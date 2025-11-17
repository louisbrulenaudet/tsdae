import ssl

from tsdae.exceptions import SSLContextError


def create_unverified_https_context() -> None:
    """
    Create an unverified HTTPS context if available.

    This function checks for the availability of an unverified HTTPS context and sets it as the default HTTPS context
    if it's available. If not available, it maintains the default behavior.

    Raises:
        SSLContextError: If an error occurs during the creation of the unverified HTTPS context.

    Note:
        This functionality is primarily used to handle SSL certificate verification in HTTPS connections, allowing
        for unverified contexts when necessary.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context

    except AttributeError as e:
        raise SSLContextError(e) from e
