"""
Logging utilities for AI Assistant Pro

Provides structured logging with different levels and output formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        use_rich: Use rich formatting for console output

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("ai_assistant_pro")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler: logging.Handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance

    Args:
        name: Logger name (uses ai_assistant_pro if None)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"ai_assistant_pro.{name}")
    return logging.getLogger("ai_assistant_pro")
