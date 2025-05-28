import logging
from typing import Literal, Optional

# Logs priority levels:
# CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET

# Global cache for logger instances:
_logger_instances = {}


def get_logger(
    name: str = "unset",
    log_to_console: bool = False,
    log_to_file: bool = True,
    log_file: Optional[str] = "app.log",
) -> logging.Logger:
    """Get logger with the given name and configuration.
    - If a logger with the same name already exists, it returns the cached instance.
    - Else, it creates a new logger with the specified configuration.

    Args:
        name (str): Name of the logger (will be logged as source).
        log_to_console (bool): Whether to log to console.
        log_to_file (bool): Whether to log to file.
        log_file (Optional[str]): Path to the log file. If None, file logging is disabled.

    Returns:
        logging.Logger: Configured logger instance.
    """

    if name in _logger_instances:
        return _logger_instances[name]

    # Create logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent duplicate logs from root logger

    formatter = logging.Formatter(
        # '%(asctime)s.%(msecs)03d [%(levelname)8s] [%(name)s] %(message)s',
        '%(asctime)s.%(msecs)03d [%(levelname)-8s] [%(name)s] %(message)s',
        datefmt='%d-%m-%y %H:%M:%S'
    )

    # File Handler
    if log_to_file and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(
        f"Logger initialized. [File: '{log_file if log_to_file else 'No'}', Console: '{'Yes' if log_to_console else 'No'}']"
    )

    _logger_instances[name] = logger
    return logger


def log_message(
    logger: logging.Logger,
    message: str,
    level: Literal["debug", "info", "warning", "error", "critical"] = "info"
):
    """Utility to log messages at various levels.

    Args:
        logger (logging.Logger): The logger instance to use.
        message (str): The message to log.
        level (Literal): The logging level to use. Defaults to 'info'.
    """

    # Same as logger.debug(message),... etc.
    getattr(logger, level)(message)


# Example usage:
if __name__ == "__main__":
    logger = get_logger(name="Test", log_to_console=True, log_to_file=True)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
