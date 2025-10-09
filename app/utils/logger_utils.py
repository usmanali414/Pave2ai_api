from loguru import logger
import sys
import os

class LogConfig:
    LOGGING_LEVEL = "DEBUG"
    LOGGING_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>"
    LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
    LOG_FILE_PATH = os.path.join(LOGS_DIR, "app.log")
    LOG_TO_FILE = "true"

    # Dictionary to keep track of custom handler IDs.
    _custom_handlers = {}

    @staticmethod
    def configure_global_logging():
        LogConfig._ensure_logs_dir()  # Ensure logs directory exists
        logger.remove()  # Remove all existing handlers

        # Configure console logging
        logger.add(
            sys.stderr,
            format=LogConfig.LOGGING_FORMAT,
            level=LogConfig.LOGGING_LEVEL,
        )

        # Configure app.log (default) logging
        if LogConfig.LOG_TO_FILE.lower() == "true":
            logger.add(
                LogConfig.LOG_FILE_PATH,
                rotation="10 MB",
                retention="30 days",
                format=LogConfig.LOGGING_FORMAT,
                level=LogConfig.LOGGING_LEVEL,
                filter=lambda record: "logger_type" not in record["extra"],  # Only log messages without logger_type
                mode="a"  # Append mode
            )

    @staticmethod
    def _ensure_logs_dir():
        """Ensure the logs directory exists"""
        os.makedirs(LogConfig.LOGS_DIR, exist_ok=True)

LogConfig.configure_global_logging()

if __name__ == "__main__":
    LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    LOG_FILE_PATH = os.path.join(LOGS_DIR, "app.log")

    print(LOG_FILE_PATH)
