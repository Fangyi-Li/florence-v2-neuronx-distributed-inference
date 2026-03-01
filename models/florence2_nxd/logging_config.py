"""
Logging configuration for Florence-2 NxD Inference integration.

This module provides centralized logging configuration for debugging and monitoring
of the Florence-2 NxD Inference implementation.
"""

import logging
import sys
from typing import Optional


# Default log format
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(message)s"
)

# Simplified format for console output
CONSOLE_LOG_FORMAT = "%(levelname)s - %(name)s - %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration for Florence-2 NxD Inference.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file. If None, only console logging is used
        console: Whether to enable console logging
        format_string: Custom format string. If None, uses default format
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging(level=logging.DEBUG, log_file="florence2.log")
        >>> logger.info("Model initialized successfully")
    """
    # Get root logger for florence2_nxd package
    logger = logging.getLogger("florence2_nxd")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Use provided format or default
    fmt = format_string or DEFAULT_LOG_FORMAT
    formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Processing vision stage 0")
    """
    return logging.getLogger(f"florence2_nxd.{name}")


# Module-level logger for this package
_package_logger: Optional[logging.Logger] = None


def init_package_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Initialize package-level logging.
    
    This should be called once at the start of the application.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Example:
        >>> from models.florence2_nxd.logging_config import init_package_logging
        >>> init_package_logging(level=logging.DEBUG, log_file="florence2_nxd.log")
    """
    global _package_logger
    _package_logger = setup_logging(level=level, log_file=log_file)


def get_package_logger() -> logging.Logger:
    """
    Get the package-level logger.
    
    If logging hasn't been initialized, it will be initialized with default settings.
    
    Returns:
        Package logger instance
    """
    global _package_logger
    if _package_logger is None:
        _package_logger = setup_logging()
    return _package_logger
