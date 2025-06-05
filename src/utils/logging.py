"""
logging.py

This module provides a helper function to set up logging for agentic projects. 
It is designed for beginners to easily get readable logs in the console and in a file, making it easier to debug and understand agent behavior.

Key Concepts:
- Logger: Records messages about what your code is doing.
- Console and file output: See logs in your terminal and save them for later review.
- Log levels: Control how much detail you see (DEBUG, INFO, etc.).

Example usage:
    from logging import get_logger
    logger = get_logger("MyAgent")
    logger.info("Agent started!")
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured to output logs to the console and to a file (with timestamp and log level).
    Log files are stored in the 'logs' directory under the project root, one file per day.
    The log level is read from the LOG_LEVEL environment variable (default: DEBUG).
    """
    logger = logging.getLogger(name)
    log_level_str = os.getenv("LOG_LEVEL", "DEBUG").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    if not logger.handlers:
        # Console handler (simple output)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Ensure logs directory exists at project root (parent of 'src')
        src_dir = os.path.dirname(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(src_dir, os.pardir))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "agentic.log")

        # File handler (one file per day)
        file_handler = TimedRotatingFileHandler(log_path, when="midnight", backupCount=14, encoding="utf-8")
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger

# Example color usage in agents:
# logger.info(f"{fore.GREEN}{style.BRIGHT}Message{style.RESET}") 