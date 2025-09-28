import logging

def get_logger(name: str = 'Tech_Challenge_3'):
    """
    Returns a logger instance.

    Args:
        name (str): The name for the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    # Basic configuration to ensure logs are displayed
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

