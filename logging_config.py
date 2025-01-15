import logging

def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set base level to DEBUG

    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    # StreamHandler for terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_format)

    # FileHandler for debug.log
    file_handler = logging.FileHandler("debug.log", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.debug("Logger setup complete.")
    return logger

