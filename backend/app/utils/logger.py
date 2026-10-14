"""
Logging configuration module
Provides unified log management, outputting to both console and file
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


def _ensure_utf8_stdout():
    """
    Ensure stdout/stderr use UTF-8 encoding
    Fixes garbled Chinese output in the Windows console
    """
    if sys.platform == "win32":
        # Reconfigure standard output to UTF-8 on Windows
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Log directory
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
)


def setup_logger(name: str = "mirofish", level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up a logger

    Args:
        name: Logger name
        level: Log level

    Returns:
        Configured logger
    """
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent logs from propagating to the root logger to avoid duplicate output
    logger.propagate = False

    # If handlers already exist, do not add duplicates
    if logger.handlers:
        return logger

    # Log format
    detailed_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )

    # 1. File handler - detailed logs (named by date, with rotation)
    log_filename = datetime.now().strftime("%Y-%m-%d") + ".log"
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_filename),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # 2. Console handler - concise logs (INFO and above)
    # Ensure UTF-8 encoding on Windows to avoid garbled Chinese characters
    _ensure_utf8_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "mirofish") -> logging.Logger:
    """
    Get a logger (create one if it does not exist)

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def run_log_prefix(
    simulation_id: Optional[str], run_id: Optional[str] = None, area: Optional[str] = None
) -> str:
    """
    Build the standard run-scoped log prefix, e.g. "[sim_x][run_y][twitter]"

    Args:
        simulation_id: Simulation ID
        run_id: Run (execution) ID
        area: Optional area tag (twitter, reddit, report, ...)

    Returns:
        Prefix string
    """
    parts = [simulation_id or "sim_?", run_id or "run_?"]
    if area:
        parts.append(area)
    return "".join(f"[{p}]" for p in parts)


class RunLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes every message with the run-scoped prefix"""

    def process(self, msg, kwargs):
        return f"{self.extra['prefix']} {msg}", kwargs


def get_run_logger(
    name: str,
    simulation_id: Optional[str],
    run_id: Optional[str] = None,
    area: Optional[str] = None,
) -> RunLoggerAdapter:
    """
    Get a logger whose messages always carry simulation_id and run_id

    Args:
        name: Logger name (e.g. 'mirofish.simulation_runner')
        simulation_id: Simulation ID
        run_id: Run (execution) ID
        area: Optional area tag

    Returns:
        RunLoggerAdapter
    """
    return RunLoggerAdapter(
        get_logger(name), {"prefix": run_log_prefix(simulation_id, run_id, area)}
    )


# Create default logger
logger = setup_logger()


# Convenience methods
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)
