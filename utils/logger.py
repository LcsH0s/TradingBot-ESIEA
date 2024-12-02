import logging
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from logging.handlers import RotatingFileHandler
import sys

# ANSI escape sequences for colors
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[35m',  # Magenta
    'RESET': '\033[0m'       # Reset
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Save original levelname
        orig_levelname = record.levelname
        # Add colors to levelname
        record.levelname = f"{COLORS.get(record.levelname, '')}{record.levelname}{COLORS['RESET']}"
        # Format the message
        result = super().format(record)
        # Restore original levelname
        record.levelname = orig_levelname
        return result

class ClassLogger(logging.LoggerAdapter):
    """Logger adapter that adds class name to messages."""
    def __init__(self, logger: logging.Logger, class_name: str):
        super().__init__(logger, {})
        self.class_name = class_name

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process the logging message and keyword arguments."""
        return f"[{self.class_name}] {msg}", kwargs

def get_class_logger(logger: logging.Logger, class_name: str) -> logging.LoggerAdapter:
    """Create a logger that includes the class name in its messages."""
    return ClassLogger(logger, class_name)

def setup_logger(name: str = 'trading_bot', level: int = logging.INFO) -> logging.Logger:
    """Setup and return a colored logger that logs to both file and console."""
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers = []
    
    logger.setLevel(level)
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger
