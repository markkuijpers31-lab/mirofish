"""
Persona identifier normalization (canonical IDs such as "C01" across name variants)
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Leading identifier token such as "C01", "C-01", "P3", "[C01]" followed by a
# separator or end of string. Letters are limited to 1-4 to avoid treating
# normal words ("Agent_5") as identifiers.
_PREFIX_ID_RE = re.compile(
    r"^\s*\[?([A-Za-z]{1,4})[-_]?(\d{1,4})\]?(?=$|[\s:.,;()/|\-–—_])"
)

# Identifier tokens inside free text (upper-case letters only, to keep noise low)
_TEXT_ID_RE = re.compile(r"\b([A-Z]{1,4})[-_]?(\d{1,4})\b")

# Ranges such as "C01-C10", "C01 – C10", "C01 to C10", "C01..C10"
_RANGE_RE = re.compile(
    r"\b([A-Z]{1,4})[-_]?(\d{1,4})\s*(?:-|–|—|\.\.|to|t/m|through|until)\s*\1?[-_]?(\d{1,4})\b"
)

# Profile/entity keys that carry an explicit identifier
EXPLICIT_ID_KEYS = ("canonical_id", "persona_id", "persona_code", "external_id", "id_code", "code")


def _format_id(letters: str, digits: str) -> str:
    return f"{letters.upper()}{digits}"


def _id_key(letters: str, digits: str) -> Tuple[str, int]:
    """Comparison key: letter family + numeric value (so C1 == C01)"""
    return letters.upper(), int(digits)


def parse_identifier(token: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Parse a leading identifier token from a name

    Args:
        token: A persona name or identifier, e.g. "C01 Dutch first-time family"

    Returns:
        (letters, digits) tuple or None
    """
    if not token:
        return None
    match = _PREFIX_ID_RE.match(str(token))
    if not match:
        return None
    return match.group(1), match.group(2)


def normalize_identifier(token: Optional[str]) -> Optional[str]:
    """
    Normalize an identifier token ("c-01" -> "C01")

    Args:
        token: Identifier token

    Returns:
        Normalized identifier or None if the token is not identifier-like
    """
    parsed = parse_identifier(token)
    return _format_id(*parsed) if parsed else None


def expand_id_ranges(text: str) -> List[str]:
    """
    Expand identifier ranges in free text ("C01-C10" -> C01..C10), keeping zero padding

    Args:
        text: Free text (e.g. an assessment brief)

    Returns:
        Ordered list of unique identifiers
    """
    ids: List[str] = []
    for match in _RANGE_RE.finditer(text or ""):
        letters, start, end = match.group(1), match.group(2), match.group(3)
        lo, hi = int(start), int(end)
        if hi < lo or hi - lo > 200:
            continue
        width = len(start)
        for n in range(lo, hi + 1):
            candidate = _format_id(letters, str(n).zfill(width))
            if candidate not in ids:
                ids.append(candidate)
    return ids


def extract_ids_from_text(text: str) -> List[str]:
    """
    Find identifier tokens (single and ranges) in free text

    Args:
        text: Free text

    Returns:
        Ordered list of unique identifiers
    """
    ids = expand_id_ranges(text)
    for match in _TEXT_ID_RE.finditer(text or ""):
        candidate = _format_id(match.group(1), match.group(2))
        if candidate not in ids:
            ids.append(candidate)
    return ids


def _explicit_id(record: Dict[str, Any]) -> Optional[str]:
    for key in EXPLICIT_ID_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


class PersonaIndex:
    """
    Maps simulation agents to canonical persona identities

    Each persona keeps:
        canonical_id: stable identifier used for reporting/funnel analysis (e.g. "C01")
        display_name: most descriptive name seen (e.g. "C01 Dutch first-time family")
        names: every name variant seen
        agent_ids: every simulation agent id that maps to this persona
    """

    def __init__(self, known_ids: Optional[Iterable[str]] = None):
        self.personas: Dict[str, Dict[str, Any]] = {}
        self.agent_to_canonical: Dict[int, str] = {}
        self.known_ids: List[str] = []
        self._known_by_key: Dict[Tuple[str, int], str] = {}
        self.accepted_families: Set[str] = set()
        for kid in known_ids or []:
            self.add_known_id(kid)

    def add_known_id(self, identifier: str):
        """Register an identifier requested by the assessment brief"""
        parsed = parse_identifier(identifier)
        if not parsed:
            return
        canonical = _format_id(*parsed)
        key = _id_key(*parsed)
        if key not in self._known_by_key:
            self._known_by_key[key] = canonical
            self.known_ids.append(canonical)
        self.accepted_families.add(parsed[0].upper())

    def canonical_for_name(self, name: Optional[str], explicit_id: Optional[str] = None) -> str:
        """
        Resolve the canonical identifier for a name

        Priority: explicit identifier > requested (known) ids > accepted id family > name itself

        Args:
            name: Persona/agent name
            explicit_id: Explicit identifier field, if any

        Returns:
            Canonical identifier
        """
        for token in (explicit_id, name):
            parsed = parse_identifier(token)
            if not parsed:
                continue
            key = _id_key(*parsed)
            if key in self._known_by_key:
                return self._known_by_key[key]
            if token is explicit_id or parsed[0].upper() in self.accepted_families:
                return _format_id(*parsed)
        if explicit_id:
            return explicit_id
        return (name or "").strip() or "unknown"

    def add_agent(self, agent: Dict[str, Any]) -> str:
        """
        Add one simulation agent

        Args:
            agent: {agent_id, name, entity_type?, ...profile fields}

        Returns:
            canonical_id assigned to the agent
        """
        name = (agent.get("name") or "").strip()
        canonical = self.canonical_for_name(name, _explicit_id(agent))
        persona = self.personas.setdefault(canonical, {
            "canonical_id": canonical,
            "display_name": name or canonical,
            "names": [],
            "agent_ids": [],
            "entity_types": [],
            "profiles": [],
        })
        if name and name not in persona["names"]:
            persona["names"].append(name)
        # Prefer the most descriptive name as display name
        if len(name) > len(persona["display_name"] or ""):
            persona["display_name"] = name
        agent_id = agent.get("agent_id")
        if agent_id is not None:
            try:
                agent_id = int(agent_id)
            except (TypeError, ValueError):
                pass
            if agent_id not in persona["agent_ids"]:
                persona["agent_ids"].append(agent_id)
            self.agent_to_canonical[agent_id] = canonical
        entity_type = agent.get("entity_type")
        if entity_type and entity_type not in persona["entity_types"]:
            persona["entity_types"].append(entity_type)
        profile = {k: v for k, v in agent.items() if k not in ("agent_id",)}
        persona["profiles"].append(profile)
        return canonical

    def resolve(self, agent_id: Any = None, agent_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Resolve an action's actor to (canonical_id, display_name)

        The agent id is the reliable link between config, profiles and action logs;
        the name is only used when the id is unknown.
        """
        try:
            agent_id = int(agent_id) if agent_id is not None else None
        except (TypeError, ValueError):
            pass
        canonical = self.agent_to_canonical.get(agent_id) if agent_id is not None else None
        if canonical is None:
            canonical = self.canonical_for_name(agent_name)
        persona = self.personas.get(canonical)
        display = persona["display_name"] if persona else (agent_name or canonical)
        return canonical, display

    def to_list(self) -> List[Dict[str, Any]]:
        """Serializable persona list"""
        return [
            {
                "canonical_id": p["canonical_id"],
                "display_name": p["display_name"],
                "names": p["names"],
                "agent_ids": p["agent_ids"],
                "entity_types": p["entity_types"],
            }
            for p in self.personas.values()
        ]


def build_persona_index(
    agents: List[Dict[str, Any]], known_ids: Optional[Iterable[str]] = None
) -> PersonaIndex:
    """
    Build a PersonaIndex from simulation agents

    An identifier family (the letter prefix, e.g. "C") is accepted when the brief
    requests it, when an agent has an explicit identifier, or when at least two
    distinct agents carry identifiers of the same family. This keeps names such as
    "GPT4 enthusiast" from being mistaken for identifiers.

    Args:
        agents: List of {agent_id, name, entity_type?, ...}
        known_ids: Identifiers requested by the assessment brief

    Returns:
        PersonaIndex
    """
    index = PersonaIndex(known_ids)

    family_members: Dict[str, Set[int]] = {}
    for agent in agents:
        explicit = _explicit_id(agent)
        if explicit and parse_identifier(explicit):
            index.accepted_families.add(parse_identifier(explicit)[0].upper())
        parsed = parse_identifier(agent.get("name"))
        if parsed:
            family_members.setdefault(parsed[0].upper(), set()).add(int(parsed[1]))
    for family, numbers in family_members.items():
        if len(numbers) >= 2:
            index.accepted_families.add(family)

    for agent in agents:
        index.add_agent(agent)
    return index
