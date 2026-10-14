"""
Action logger.
Used to log each agent's actions in OASIS simulations for backend monitoring.

Log structure:
    sim_xxx/
    ├── twitter/
    │   └── actions.jsonl    # Twitter platform action log
    ├── reddit/
    │   └── actions.jsonl    # Reddit platform action log
    ├── simulation.log       # Main simulation process log
    └── run_state.json       # Run state (for API queries)
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Execution identity, injected by the backend runner. Every log line carries the
# run id so a log file can always be checked for evidence of another execution.
RUN_ID = os.environ.get("MIROFISH_RUN_ID")
SIMULATION_ID = os.environ.get("MIROFISH_SIMULATION_ID")


def _write_entry(log_path: str, entry: Dict[str, Any]):
    """Append one JSON line, tagged with the run id"""
    if RUN_ID:
        entry["run_id"] = RUN_ID
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class PlatformActionLogger:
    """Single-platform action logger"""

    def __init__(self, platform: str, base_dir: str):
        """
        Initialize the logger.

        Args:
            platform: Platform name (twitter/reddit)
            base_dir: Base path of the simulation directory
        """
        self.platform = platform
        self.base_dir = base_dir
        self.log_dir = os.path.join(base_dir, platform)
        self.log_path = os.path.join(self.log_dir, "actions.jsonl")
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure the directory exists"""
        os.makedirs(self.log_dir, exist_ok=True)

    def log_action(
        self,
        round_num: int,
        agent_id: int,
        agent_name: str,
        action_type: str,
        action_args: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None,
        success: bool = True,
        origin: str = "agent_generated",
        origin_flag: Optional[str] = None,
    ):
        """
        Log an action

        origin: agent_generated (the agent's own LLM step), system_seed (initial
        posts published by the system) or researcher_instruction (content that
        reproduces a private instruction). Only agent_generated is behavioural evidence.
        """
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "agent_name": agent_name,
            "action_type": action_type,
            "action_args": action_args or {},
            "result": result,
            "success": success,
            "origin": origin,
        }
        if origin_flag:
            entry["origin_flag"] = origin_flag

        _write_entry(self.log_path, entry)

    def log_event(self, event_type: str, round_num: Optional[int] = None, origin: str = "orchestration", **fields):
        """
        Log a non-action event (private instruction delivery, stage completion,
        rejected seed post). Events are inspectable but never counted as actions.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "origin": origin,
            "platform": self.platform,
        }
        if round_num is not None:
            entry["round"] = round_num
        entry.update({k: v for k, v in fields.items() if v is not None})
        _write_entry(self.log_path, entry)

    def log_round_start(self, round_num: int, simulated_hour: int):
        """Log round start"""
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "event_type": "round_start",
            "simulated_hour": simulated_hour,
        }

        _write_entry(self.log_path, entry)

    def log_round_end(self, round_num: int, actions_count: int):
        """Log round end"""
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "event_type": "round_end",
            "actions_count": actions_count,
        }

        _write_entry(self.log_path, entry)

    def log_simulation_start(self, config: Dict[str, Any]):
        """Log simulation start"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "simulation_start",
            "platform": self.platform,
            "total_rounds": config.get("time_config", {}).get(
                "total_simulation_hours", 72
            )
            * 2,
            "agents_count": len(config.get("agent_configs", [])),
        }

        _write_entry(self.log_path, entry)

    def log_simulation_end(self, total_rounds: int, total_actions: int):
        """Log simulation end"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "simulation_end",
            "platform": self.platform,
            "total_rounds": total_rounds,
            "total_actions": total_actions,
        }

        _write_entry(self.log_path, entry)


class SimulationLogManager:
    """
    Simulation log manager.
    Unified management of all log files, separated by platform.
    """

    def __init__(self, simulation_dir: str):
        """
        Initialize the log manager.

        Args:
            simulation_dir: Simulation directory path
        """
        self.simulation_dir = simulation_dir
        self.twitter_logger: Optional[PlatformActionLogger] = None
        self.reddit_logger: Optional[PlatformActionLogger] = None
        self._main_logger: Optional[logging.Logger] = None

        # Set up main log
        self._setup_main_logger()

    def _setup_main_logger(self):
        """Set up the main simulation log"""
        log_path = os.path.join(self.simulation_dir, "simulation.log")

        # Create logger
        self._main_logger = logging.getLogger(
            f"simulation.{os.path.basename(self.simulation_dir)}"
        )
        self._main_logger.setLevel(logging.INFO)
        self._main_logger.handlers.clear()

        # File handler. Append: the runner already redirects the process output
        # into the same file, and every run has its own directory.
        prefix = f"[{SIMULATION_ID or 'sim_?'}][{RUN_ID or 'run_?'}] " if RUN_ID else ""
        file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s - %(levelname)s - {prefix}%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._main_logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        self._main_logger.addHandler(console_handler)

        self._main_logger.propagate = False

    def get_twitter_logger(self) -> PlatformActionLogger:
        """Get the Twitter platform logger"""
        if self.twitter_logger is None:
            self.twitter_logger = PlatformActionLogger("twitter", self.simulation_dir)
        return self.twitter_logger

    def get_reddit_logger(self) -> PlatformActionLogger:
        """Get the Reddit platform logger"""
        if self.reddit_logger is None:
            self.reddit_logger = PlatformActionLogger("reddit", self.simulation_dir)
        return self.reddit_logger

    def log(self, message: str, level: str = "info"):
        """Log a main log message"""
        if self._main_logger:
            getattr(self._main_logger, level.lower(), self._main_logger.info)(message)

    def info(self, message: str):
        self.log(message, "info")

    def warning(self, message: str):
        self.log(message, "warning")

    def error(self, message: str):
        self.log(message, "error")

    def debug(self, message: str):
        self.log(message, "debug")


# ============ Legacy interface compatibility ============


class ActionLogger:
    """
    Action logger (legacy interface compatible).
    Using SimulationLogManager is recommended instead.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._ensure_dir()

    def _ensure_dir(self):
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_action(
        self,
        round_num: int,
        platform: str,
        agent_id: int,
        agent_name: str,
        action_type: str,
        action_args: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None,
        success: bool = True,
    ):
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "platform": platform,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "action_type": action_type,
            "action_args": action_args or {},
            "result": result,
            "success": success,
        }

        _write_entry(self.log_path, entry)

    def log_round_start(self, round_num: int, simulated_hour: int, platform: str):
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "platform": platform,
            "event_type": "round_start",
            "simulated_hour": simulated_hour,
        }

        _write_entry(self.log_path, entry)

    def log_round_end(self, round_num: int, actions_count: int, platform: str):
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "platform": platform,
            "event_type": "round_end",
            "actions_count": actions_count,
        }

        _write_entry(self.log_path, entry)

    def log_simulation_start(self, platform: str, config: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform,
            "event_type": "simulation_start",
            "total_rounds": config.get("time_config", {}).get(
                "total_simulation_hours", 72
            )
            * 2,
            "agents_count": len(config.get("agent_configs", [])),
        }

        _write_entry(self.log_path, entry)

    def log_simulation_end(self, platform: str, total_rounds: int, total_actions: int):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform,
            "event_type": "simulation_end",
            "total_rounds": total_rounds,
            "total_actions": total_actions,
        }

        _write_entry(self.log_path, entry)


# Global logger instance (legacy interface compatible)
_global_logger: Optional[ActionLogger] = None


def get_logger(log_path: Optional[str] = None) -> ActionLogger:
    """Get the global logger instance (legacy interface compatible)"""
    global _global_logger

    if log_path:
        _global_logger = ActionLogger(log_path)

    if _global_logger is None:
        _global_logger = ActionLogger("actions.jsonl")

    return _global_logger
