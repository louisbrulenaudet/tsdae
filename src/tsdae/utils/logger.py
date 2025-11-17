import logging
import sys


def get_logger(
    name: str | None = None,
    level: int = logging.ERROR,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Get or create a logger with standardized configuration.

    This function follows industry best practices for Python logging:
    - Uses module-level logger configuration
    - Avoids calling basicConfig multiple times
    - Provides consistent formatting across the application
    - Allows customization of log level and format

    Args:
        name: The name of the logger. If None, uses the calling module's name.
            Defaults to None.
        level: The logging level to set. Defaults to logging.ERROR.
        format_string: Custom format string for log messages. If None, uses a
            standard format. Defaults to None.

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.error("An error occurred")
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get logger with the specified name
    logger = logging.getLogger(name or __name__)

    # Only configure if the logger doesn't have handlers yet
    # This prevents duplicate handlers when called multiple times
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(format_string)

        # Add formatter to handler
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger
