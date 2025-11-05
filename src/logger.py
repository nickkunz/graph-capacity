## libraries
import os
import sys
import logging
from datetime import datetime

## logging configuration
def logging_config(
    log_level: int,
    log_to_file: bool = False,
    log_dir: str = './logs',
    log_name: str = 'graph-capacity'
    ) -> logging.Logger:
    
    """
    Desc:
        Initializes and configures a logger.
        Sets up a logger that can write to the console and optionally to a file.
        This function is designed to be idempotent, where it can be called multiple 
        times without creating duplicate handlers.

    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_to_file: If True, logs will also be written to a file.
        log_dir: The directory where log files will be stored.
        log_name: The name of the logger.

    Returns:
        A configured logging.Logger instance.
    """

    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    ## prevent adding duplicate handlers
    if logger.hasHandlers():
        return logger

    ## create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    ## console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    ## file handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{log_name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"logging to file: {log_file}")

    return logger

## execute logging
if __name__ == '__main__':
    logger = logging_config(log_level = logging.INFO, log_to_file = False)
    logger.info("Initialized logging.")