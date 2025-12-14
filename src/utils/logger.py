"""
Logging utility for ASL&AI system.

Provides structured logging with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import config


def setup_logger(
    name: str = "asl_ai",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Parameters
    ----------
    name : str
        Logger name (default: "asl_ai")
    log_level : int
        Logging level (default: INFO)
    log_file : Optional[Path]
        Path to log file. If None, only console logging is used.
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger instance.
    
    Parameters
    ----------
    name : Optional[str]
        Logger name. If None, uses "asl_ai"
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger_name = name or "asl_ai"
    logger = logging.getLogger(logger_name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        log_file = config.LOGS_DIR / f"{logger_name}_{datetime.now().strftime('%Y%m%d')}.log"
        logger = setup_logger(logger_name, log_file=log_file)
    
    return logger


