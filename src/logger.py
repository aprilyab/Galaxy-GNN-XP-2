import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "stage1", level: int = logging.INFO, log_file: str = "extraction.log") -> logging.Logger:
    """Configures and returns a logger instance with file and stream handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Stream Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # File Handler
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Global logger instance
logger = setup_logger()
