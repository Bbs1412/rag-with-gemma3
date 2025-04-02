import logging
from typing import Literal

# Logs priority levels:
# CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET


class Logger:
    def __init__(self, name: str, log_file: str = "app.log", log_to_console: bool = False, log_to_file: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file

        # Create handler for file
        # only messages with level >= DEBUG will be logged
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.DEBUG)

        # Set the logging level for console handler
        # to INFO so that only messages with level >= INFO will be logged
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%d-%m-%y %H:%M:%S'
        )

        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)

        # Add handlers to the logger
        if self.log_to_file:
            self.logger.addHandler(self.file_handler)

        if self.log_to_console:
            self.logger.addHandler(self.console_handler)

        self.logger.propagate = False

        # self.logger.info("Logger initialized.")
        # self.logger.debug("Logger initialized in debug mode.")
        # self.logger.warning("Logger initialized in warning mode.")
        # self.logger.error("Logger initialized in error mode.")
        # self.logger.critical("Logger initialized in critical mode.")

    # Function to log messages:
    def log(self, message: str, level: Literal['debug', 'info', 'warning', 'error', 'critical']):
        if level == "debug":
            self.logger.debug(f"{message}")
        elif level == "info":
            self.logger.info(f"{message}")
        elif level == "warning":
            self.logger.warning(f"{message}")
        elif level == "error":
            self.logger.error(f"{message}")
        elif level == "critical":
            self.logger.critical(f"{message}")
        else:
            raise ValueError(
                "Invalid log level. Choose from ['debug', 'info', 'warning', 'error', 'critical'].")


# Example usage:
if __name__ == "__main__":
    logger = Logger(
        name="example_logger", log_to_console=True, log_to_file=True)
    logger.log("This is a debug message.", "debug")
    logger.log("This is an info message.", "info")
    logger.log("This is a warning message.", "warning")
    logger.log("This is an error message.", "error")
    logger.log("This is a critical message.", "critical")
