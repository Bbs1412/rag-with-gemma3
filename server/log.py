import logging
from typing import Literal

# Logs priority levels:
# CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET

_loggers = {}

def get_logger(name: str = "Streamlit", **kwargs):
    if name not in _loggers:
        _loggers[name] = Logger(name=name, **kwargs)
    return _loggers[name]

class Logger:
    def __init__(self, name: str, log_file: str = "app.log", log_to_console: bool = False, log_to_file: bool = True):
        if name in _loggers:
            self.logger = _loggers[name].logger
            
        else:
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.DEBUG)
            self.log_to_console = log_to_console
            self.log_to_file = log_to_file

            # # Avoid adding handlers multiple times
            # if not self.logger.handlers:
            # File Handler
            if self.log_to_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter(
                    '%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S'
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            # Console Handler
            if self.log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S'
                )
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            self.logger.propagate = False
            _loggers[name] = self
            
            self.logger.info(
                f"Logger initialized for {name}. [File:{'Yes' if log_to_file else 'No'}, Console:{'Yes' if log_to_console else 'No'}]")

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
