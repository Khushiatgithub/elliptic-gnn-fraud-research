"""
Research Logging Utility.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "elliptic_audit", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a standardized logger that outputs to console and file.
    
    Args:
        name: Name of the logger.
        log_file: Path to log file. If None, default to results/logs/audit.log.
        level: Logging level (default: INFO).
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger was already created
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file is None:
        log_dir = Path("results/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "dataset_audit.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger
