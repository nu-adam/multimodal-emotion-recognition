import logging
import os
from datetime import datetime


def setup_logger(log_dir='results/logs', log_file='training', log_level=logging.INFO):
    """
    Set up the logging system for training and evaluation.
    
    Args:
        log_dir (str): Directory to save log files.
        log_file (str): Name of the log file (training, evaluation, etc.)
        log_level: Logging level (INFO, DEBUG, etc.)    
        
    Returns:
        logger: Configured logger instance.
    """
    # Create log directory if not exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create unique log filename with timestamp
    log_filename = os.path.join(log_dir, f'{log_file}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')

    # Create logger
    logger = logging.getLogger('FaceEmotionRecognition')
    logger.setLevel(log_level)
    
    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler: Outputs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler: Logs to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
