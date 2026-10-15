"""
Run orchestration shared by the OASIS run scripts: seed content, action origins and private experiment stages.

Pure Python (no OASIS import) so the logic is unit-testable. Every logged action
carries an origin:
    agent_generated         decided by the agent's own LLM step (behavioural evidence)
    system_seed             initial in-world posts published by the system at round 0
    researcher_instruction  content reproducing a private research instruction
    orchestration           stage bookkeeping (events only)
Only agent_generated actions may count as behavioural evidence.
"""

import json
import os
import re
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

ORIGIN_AGENT_GENERATED = "agent_generated"
ORIGIN_SYSTEM_SEED = "system_seed"
ORIGIN_RESEARCHER_INSTRUCTION = "researcher_instruction"
ORIGIN_ORCHESTRATION = "orchestration"

PLAN_FILE = "experiment_plan.json"

# Actions whose authored text is public
PUBLIC_TEXT_ACTIONS = {"CREATE_POST", "CREATE_COMMENT", "QUOTE_POST", "REPOST"}

PRIVATE_FRAME = (
    "[PRIVATE CONTEXT FOR YOU ONLY - nobody else can see this. Never quote, copy or post this text; "
    "it is not a message from another user.]\n{instruction}\n"
    "[You stay completely free in how you respond: you may ignore it, look into it, try it or reject it, "
    "exactly as your own persona would. Act only through your normal social media behaviour.]"
)
CONTEXT_FRAME = (
    "\n[Factual information you can rely on - public product documentation, not advice and not a "
    "recommendation. Information that is not listed here may still exist; do not assume something "
    "is missing just because it is not mentioned:]\n{context}"
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_INSTRUCTION_MARKERS_RE = re.compile(
    r"\b(?:opening[- ]prompt|cold[- ]start|prompt|instructions?|instructie\w*|opdracht\w*|"
    r"test[- ]task|task\s*:|stage\s*\d|fase\s*\d|researchers?\b|assessment brief|funnel stage|"
    r"hypothes[ie]s|persona\b|you are (?:a|an|the)\b|act as\b|role[- ]?play)",
    re.IGNORECASE,
)
_LEADING_ID_RE = re.compile(r"^\s*(?:[-*•]\s*)?\[?[A-Za-z]{1,4}[-_]?\d{1,4}\]?\s*[-–—:|]\s*\S")


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def overlap_ratio(text: str, reference: str, n: int = 5) -> float:
    """Share of text's n-word shingles found verbatim in reference (mirrors app.services.research_instructions)"""
    a, b = _words(text), _words(reference)
    if not a or not b:
        return 0.0
    if len(a) < n:
        return 1.0 if len(a) >= 3 and " ".join(a) in " ".join(b) else 0.0
    shingles = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    ref = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(shingles & ref) / len(shingles)


def reproduces_private_text(text: str, private_texts: Iterable[str], threshold: float = 0.5) -> bool:
    for private in private_texts or []:
        if private and (overlap_ratio(text, private) >= threshold or overlap_ratio(private, text) >= 0.8):
            return True
    return False


def load_plan(simulation_dir: str) -> Dict[str, Any]:
    """The run's resolved experiment plan (empty plan when absent)"""
    path = os.path.join(simulation_dir, PLAN_FILE)
    if not os.path.exists(path):
        return {"stages": [], "private_texts": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception:
        return {"stages": [], "private_texts": []}
    plan.setdefault("stages", [])
    plan.setdefault("private_texts", [])
    return plan


def private_texts_for_run(config: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for t in list(plan.get("private_texts") or []) + list(config.get("private_texts") or []):
        if t and t not in texts:
            texts.append(t)
    for stage in plan.get("stages") or []:
        t = stage.get("private_instruction")
        if t and t not in texts:
            texts.append(t)
    return texts


def safe_seed_posts(
    config: Dict[str, Any], private_texts: Iterable[str], blocked_terms: Iterable[str] = ()
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Seed posts that may be published (last line of defence for older configs)

    Args:
        config: simulation_config.json
        private_texts: Private instruction texts
        blocked_terms: Terms seed posts may not contain (e.g. the product under test)

    Returns:
        (posts to publish, rejected posts with reason)
    """
    blocked = [t.lower() for t in blocked_terms or [] if t]
    publish, rejected = [], []
    seen = set()
    for post in (config.get("event_config") or {}).get("initial_posts") or []:
        content = str((post or {}).get("content") or "").strip()
        if not content:
            continue
        key = (post.get("poster_agent_id"), content)
        if key in seen:
            rejected.append({**post, "reason": "duplicate seed post"})
            continue
        seen.add(key)
        hit = next((t for t in blocked if t in content.lower()), None)
        if hit:
            rejected.append({**post, "reason": f"mentions blocked term '{hit}'"})
        elif reproduces_private_text(content, private_texts):
            rejected.append({**post, "reason": "reproduces a private research instruction"})
        elif _INSTRUCTION_MARKERS_RE.search(content) or _LEADING_ID_RE.match(content):
            rejected.append({**post, "reason": "reads like a research instruction"})
        else:
            publish.append(post)
    return publish, rejected


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u3002])\s+|\n+")

PUBLIC_TEXT_WITHHELD = "public_text_withheld"
PUBLIC_TEXT_STRIPPED = "public_text_stripped"


def sanitize_public_text(text: str, private_texts: Iterable[str]) -> Tuple[str, str]:
    """
    Remove private research wording from text an agent is about to publish

    Sentences that reproduce a private instruction are dropped; when nothing
    publishable remains (or the rest still reproduces it) the text is withheld.
    Factual product context is not private and is never touched here.

    Returns:
        (text to publish, verdict) with verdict "clean" | "stripped" | "withheld"
    """
    private_texts = [p for p in private_texts or [] if p]
    if not text or not private_texts or not reproduces_private_text(text, private_texts):
        return text, "clean"
    kept = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text)
            if p.strip() and not reproduces_private_text(p, private_texts)]
    remaining = " ".join(kept).strip()
    if len(_words(remaining)) < 3 or reproduces_private_text(remaining, private_texts):
        return "", "withheld"
    return remaining, "stripped"


# OASIS platform actions that persist agent-authored public text:
# name -> (text from the action message, message with replaced text)
_GUARDED_ACTIONS = {
    "create_post": (lambda m: m, lambda m, text: text),
    "create_comment": (lambda m: m[1], lambda m, text: (m[0], text)),
    "quote_post": (lambda m: m[1], lambda m, text: (m[0], text)),
}


def install_public_text_guard(platform: Any, private_texts: Iterable[str], on_event: Any = None) -> List[str]:
    """
    Guard an OASIS platform so private instructions never reach its database

    Wraps the platform actions that write public text (post, comment, quote). The
    check runs BEFORE OASIS inserts anything: withheld content is never persisted
    (the action fails for the agent), stripped content is persisted without the
    private sentences. The wrappers keep OASIS's dispatch signature
    (self, agent_id, message).

    Args:
        platform: oasis Platform instance (env.platform)
        private_texts: Private instruction texts of the run
        on_event: Optional callable(event_type, agent_id, action, removed_chars)

    Returns:
        Names of the guarded actions
    """
    import types

    private_texts = [p for p in private_texts or [] if p]
    guarded_names = []
    if not private_texts:
        return guarded_names
    for name, (extract, rebuild) in _GUARDED_ACTIONS.items():
        original = getattr(platform, name, None)
        if original is None or getattr(original, "_mirofish_guarded", False):
            continue

        def make(original=original, name=name, extract=extract, rebuild=rebuild):
            async def guarded(self, agent_id, message):
                try:
                    text = extract(message)
                except Exception:
                    text = None
                if not isinstance(text, str):
                    return await original(agent_id, message)
                clean, verdict = sanitize_public_text(text, private_texts)
                if verdict != "clean" and on_event:
                    on_event(PUBLIC_TEXT_WITHHELD if verdict == "withheld" else PUBLIC_TEXT_STRIPPED,
                             agent_id, name, len(text) - len(clean))
                if verdict == "withheld":
                    return {"success": False, "error": "Content withheld: it reproduces private research context"}
                return await original(agent_id, rebuild(message, clean) if verdict == "stripped" else message)
            guarded._mirofish_guarded = True
            return guarded

        setattr(platform, name, types.MethodType(make(), platform))
        guarded_names.append(name)
    return guarded_names


def classify_origin(action: Dict[str, Any], private_texts: Iterable[str]) -> Tuple[str, Optional[str]]:
    """
    Origin of an action read from the OASIS trace after an LLM step

    Returns:
        (origin, flag) where flag explains a non-agent origin
    """
    if action.get("action_type") in PUBLIC_TEXT_ACTIONS:
        content = (action.get("action_args") or {}).get("content") or ""
        if content and reproduces_private_text(content, private_texts):
            return ORIGIN_RESEARCHER_INSTRUCTION, "instruction_leak"
    return ORIGIN_AGENT_GENERATED, None


CONTRACT_FILE = "run_contract.json"


def resolve_total_rounds(config: Dict[str, Any], simulation_dir: str, max_rounds: Optional[int] = None) -> Tuple[int, str]:
    """
    Number of rounds a run executes

    A contracted run (run_contract.json next to the config) executes exactly the
    contract's max_rounds; a different --max-rounds is an error. Otherwise the
    generated time_config decides, truncated by max_rounds.

    Returns:
        (total_rounds, source) with source "run_contract" | "time_config" | "max_rounds"

    Raises:
        ValueError: contract without a valid max_rounds, or a conflicting max_rounds
    """
    path = os.path.join(simulation_dir, CONTRACT_FILE)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            rounds = (json.load(f).get("contract") or {}).get("max_rounds")
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
            raise ValueError("run_contract.json has no valid max_rounds")
        if max_rounds is not None and max_rounds > 0 and max_rounds != rounds:
            raise ValueError(f"--max-rounds {max_rounds} conflicts with the Run Contract ({rounds})")
        return rounds, "run_contract"
    time_config = config.get("time_config", {})
    total_rounds = (time_config.get("total_simulation_hours", 72) * 60) // time_config.get("minutes_per_round", 30)
    if max_rounds is not None and max_rounds > 0 and max_rounds < total_rounds:
        return max_rounds, "max_rounds"
    return total_rounds, "time_config"


def max_trace_rowid(db_path: str) -> int:
    """Highest rowid of the OASIS trace table (0 when unavailable)"""
    if not os.path.exists(db_path):
        return 0
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT MAX(rowid) FROM trace").fetchone()
        finally:
            conn.close()
        return int(row[0] or 0)
    except Exception:
        return 0


class StageScheduler:
    """
    Decides which private stage instructions are delivered to which agent and when

    A stage is eligible for an agent from its trigger round on, while its
    condition holds; it is delivered once. Completion is evaluated on the
    agent's own actions after delivery. Nothing here forces any behaviour.
    """

    def __init__(self, plan: Dict[str, Any], active_agent_ids: Iterable[int]):
        self.stages = sorted(plan.get("stages") or [], key=lambda s: (s.get("order", 0), s.get("trigger_round", 1)))
        self.active_agent_ids = {int(a) for a in active_agent_ids}
        self.delivered: Dict[Tuple[str, int], int] = {}
        self.completed: Dict[Tuple[str, int], int] = {}
        self.history: Dict[int, List[Dict[str, Any]]] = {}

    # ---- history ----

    def observe(self, round_num: int, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Record logged actions and return stage completion events

        Args:
            round_num: Round the actions belong to
            actions: [{agent_id, action_type, action_args, origin}]
        """
        for a in actions:
            try:
                aid = int(a.get("agent_id"))
            except (TypeError, ValueError):
                continue
            self.history.setdefault(aid, []).append({**a, "round": round_num})
        return self._check_completion(round_num)

    def _authored(self, agent_id: int, since_round: int = -1) -> str:
        """Text the agent itself wrote or searched for (its own agent-generated actions)"""
        parts = []
        for a in self.history.get(agent_id, []):
            if a.get("round", 0) < since_round or a.get("origin", ORIGIN_AGENT_GENERATED) != ORIGIN_AGENT_GENERATED:
                continue
            args = a.get("action_args") or {}
            for key in ("content", "query"):
                if args.get(key):
                    parts.append(str(args[key]))
        return " ".join(parts).lower()

    def _texts(self, agent_id: int, since_round: int = -1) -> str:
        parts = []
        for a in self.history.get(agent_id, []):
            if a.get("round", 0) < since_round:
                continue
            args = a.get("action_args") or {}
            for key in ("content", "query", "post_content", "comment_content", "original_content", "quote_content"):
                if args.get(key):
                    parts.append(str(args[key]))
        return " ".join(parts).lower()

    @staticmethod
    def _mentions(text: str, terms: Iterable[str]) -> bool:
        return any(t and t.lower() in text for t in terms or [])

    # ---- conditions ----

    def _condition_holds(self, stage: Dict[str, Any], agent_id: int) -> bool:
        cond = stage.get("condition") or {"type": "always"}
        ctype = cond.get("type", "always")
        if ctype == "always":
            return True
        if ctype == "not_mentioned":
            return not self._mentions(self._texts(agent_id), cond.get("terms"))
        if ctype == "mentioned":
            return self._mentions(self._texts(agent_id), cond.get("terms"))
        if ctype == "authored_mentioned":
            return self._mentions(self._authored(agent_id), cond.get("terms"))
        if ctype == "authored_not_mentioned":
            return not self._mentions(self._authored(agent_id), cond.get("terms"))
        if ctype == "stage_completed":
            return (cond.get("stage_id"), agent_id) in self.completed
        if ctype == "stage_not_completed":
            return (cond.get("stage_id"), agent_id) not in self.completed
        return False

    def due(self, round_num: int) -> List[Dict[str, Any]]:
        """
        Deliveries for this round

        Returns:
            [{stage_id, stage_name, agent_id, instruction, framed}]
        """
        deliveries = []
        for stage in self.stages:
            if round_num < int(stage.get("trigger_round", 1)):
                continue
            for aid in stage.get("resolved_agent_ids") or []:
                aid = int(aid)
                key = (stage["id"], aid)
                if aid not in self.active_agent_ids or key in self.delivered:
                    continue
                if not self._condition_holds(stage, aid):
                    continue
                self.delivered[key] = round_num
                instruction = stage.get("private_instruction") or ""
                context = stage.get("context") or ""
                framed = PRIVATE_FRAME.format(instruction=instruction)
                if context:
                    framed += CONTEXT_FRAME.format(context=context)
                deliveries.append({
                    "stage_id": stage["id"],
                    "stage_name": stage.get("name") or stage["id"],
                    "agent_id": aid,
                    "instruction": instruction,
                    "context_chars": len(context),
                    "records": list(stage.get("records") or []),
                    "framed": framed,
                })
                if (stage.get("completion") or {}).get("type", "on_delivery") == "on_delivery":
                    self.completed[key] = round_num
        return deliveries

    def _check_completion(self, round_num: int) -> List[Dict[str, Any]]:
        events = []
        for stage in self.stages:
            comp = stage.get("completion") or {"type": "on_delivery"}
            ctype = comp.get("type", "on_delivery")
            for (sid, aid), delivered_round in list(self.delivered.items()):
                if sid != stage["id"] or (sid, aid) in self.completed:
                    continue
                own = [a for a in self.history.get(aid, [])
                       if a.get("round", 0) >= delivered_round and a.get("origin", ORIGIN_AGENT_GENERATED) == ORIGIN_AGENT_GENERATED]
                done = False
                if ctype == "any_action":
                    done = bool(own)
                elif ctype == "action_types":
                    done = any(a.get("action_type") in (comp.get("action_types") or []) for a in own)
                elif ctype == "mentioned":
                    done = self._mentions(self._texts(aid, delivered_round), comp.get("terms"))
                elif ctype == "authored_mentioned":
                    done = self._mentions(self._authored(aid, delivered_round), comp.get("terms"))
                elif ctype == "rounds_elapsed":
                    done = round_num - delivered_round >= int(comp.get("rounds", 1))
                if done:
                    self.completed[(sid, aid)] = round_num
                    events.append({"stage_id": sid, "stage_name": stage.get("name") or sid, "agent_id": aid,
                                   "delivered_round": delivered_round, "completed_round": round_num})
        return events

    def summary(self) -> Dict[str, Any]:
        return {
            "delivered": [{"stage_id": s, "agent_id": a, "round": r} for (s, a), r in sorted(self.delivered.items())],
            "completed": [{"stage_id": s, "agent_id": a, "round": r} for (s, a), r in sorted(self.completed.items())],
        }


def deliver_private_instruction(agent: Any, framed_text: str) -> bool:
    """
    Put a private instruction into an OASIS agent's memory (never on the platform)

    Returns:
        True when delivered
    """
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole

    message = BaseMessage.make_user_message(role_name="Private context", content=framed_text)
    agent.update_memory(message, OpenAIBackendRole.USER)
    return True


class RunOrchestrator:
    """Per-platform glue used by the run scripts"""

    def __init__(self, config: Dict[str, Any], simulation_dir: str, action_logger: Any = None):
        self.config = config
        self.plan = load_plan(simulation_dir)
        self.private_texts = private_texts_for_run(config, self.plan)
        agent_ids = [int(a.get("agent_id")) for a in config.get("agent_configs", []) if a.get("agent_id") is not None]
        self.scheduler = StageScheduler(self.plan, agent_ids)
        self.action_logger = action_logger
        self.agent_names = {int(a["agent_id"]): a.get("entity_name") for a in config.get("agent_configs", [])
                            if a.get("agent_id") is not None}
        self.current_round = 0

    def seed_posts(self) -> List[Dict[str, Any]]:
        posts, rejected = safe_seed_posts(self.config, self.private_texts, self.plan.get("seed_blocked_terms"))
        if self.action_logger:
            for post in rejected:
                self.action_logger.log_event(
                    "seed_post_rejected", round_num=0, origin=ORIGIN_ORCHESTRATION,
                    agent_id=post.get("poster_agent_id"), content=post.get("content"), reason=post.get("reason"),
                )
        return posts

    def deliver_due(self, round_num: int, get_agent) -> List[int]:
        """
        Deliver due private instructions; returns agent ids that must act this round

        Args:
            round_num: Current round (>= 1)
            get_agent: callable(agent_id) -> OASIS agent
        """
        self.current_round = round_num
        activated = []
        for d in self.scheduler.due(round_num):
            status, error = "delivered", None
            try:
                deliver_private_instruction(get_agent(d["agent_id"]), d["framed"])
                activated.append(d["agent_id"])
            except Exception as e:  # never fall back to posting
                status, error = "failed", str(e)
            if self.action_logger:
                self.action_logger.log_event(
                    "private_instruction", round_num=round_num, origin=ORIGIN_RESEARCHER_INSTRUCTION,
                    agent_id=d["agent_id"], agent_name=self.agent_names.get(d["agent_id"]),
                    stage_id=d["stage_id"], stage_name=d["stage_name"], instruction=d["instruction"],
                    context_chars=d["context_chars"], records=d["records"] or None,
                    delivery="agent_memory", status=status, error=error,
                )
        return activated

    def guard_platform(self, platform: Any) -> List[str]:
        """Install the pre-persistence leak guard on an OASIS platform (see install_public_text_guard)"""

        def on_event(event_type, agent_id, action, removed_chars):
            if self.action_logger:
                # The withheld text itself is never logged
                self.action_logger.log_event(
                    event_type, round_num=self.current_round, origin=ORIGIN_ORCHESTRATION,
                    agent_id=agent_id, agent_name=self.agent_names.get(agent_id), action=action,
                    removed_chars=removed_chars,
                )

        return install_public_text_guard(platform, self.private_texts, on_event)

    def classify(self, action: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        return classify_origin(action, self.private_texts)

    def observe(self, round_num: int, actions: List[Dict[str, Any]]):
        for event in self.scheduler.observe(round_num, actions):
            if self.action_logger:
                self.action_logger.log_event("stage_completed", round_num=round_num, origin=ORIGIN_ORCHESTRATION, **event)
