from __future__ import annotations

from typing import Any

from tsdae.enums import ErrorCodes

__all__ = [
    "CoreError",
    "SSLContextError",
    "NLTKDownloadError",
    "InvalidDatasetInputError",
    "DatasetLoadError",
    "InvalidDatasetTypeError",
    "InvalidRegexPatternError",
    "DatasetConstraintError",
    "InvalidSentencesInputError",
    "InvalidBatchSizeError",
    "InvalidNumWorkersError",
    "DataLoaderCreationError",
    "InvalidModelNameError",
    "ModelCreationError",
    "InvalidModelInputError",
    "InvalidTieEncoderDecoderError",
    "LossCreationError",
    "TrainingError",
    "ValidationError",
    "TypeMismatchError",
]

import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class CoreError(Exception):
    """
    A custom exception class for handling application-specific errors.

    This exception includes an error message, an error code, and optional details.
    It also logs the error upon initialization.

    Attributes:
        message (str): A descriptive error message.
        code (ErrorCodes): An enumerated error code representing the specific error type.
        details (dict[str, Any] | str | None): Additional details about the error.
    """

    message: str
    code: ErrorCodes
    details: dict[str, Any] | str | None

    def __init__(
        self,
        message: str,
        code: ErrorCodes,
        details: dict[str, Any] | str | None = None,
    ) -> None:
        """
        Initialize a CoreError instance with an error message, code, and optional details.

        The error is logged automatically when an instance is created.

        Args:
            message (str): The error message.
            code (ErrorCodes): A predefined error code representing the error type.
            details (dict[str, Any] | str | None, optional): Additional information about the error.
                Can be a dictionary, a string, or None. Defaults to None.
        """
        self.message = message
        self.code = code
        self.details = details

        # Log the error with its details
        logger.error(
            f"CoreError: {self.message} [Code: {self.code}] Details: {self.details}"
        )

    def __str__(self) -> str:
        """
        Return a string representation of the error, including the message, code, and optional details.

        Returns:
            str: A formatted string describing the error.

        Example:
            >>> error = CoreError("Invalid input", ErrorCodes.INVALID_INPUT, {"field": "email"})
            >>> print(str(error))
            "CoreError: Invalid input [Code: INVALID_INPUT] Details: {'field': 'email'}"
        """
        detail_part = f" Details: {self.details}" if self.details else ""
        return f"{self.__class__.__name__}: {self.message} [Code: {self.code}]{detail_part}"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the CoreError instance into a dictionary format.

        This is useful for structured logging or returning errors in API responses.

        Returns:
            dict[str, Any]: A dictionary containing error details.

        Example:
            >>> error = CoreError("Access denied", ErrorCodes.PERMISSION_DENIED, "User lacks admin rights")
            >>> error.to_dict()
            {
                "error": "CoreError",
                "message": "Access denied",
                "code": "PERMISSION_DENIED",
                "details": "User lacks admin rights"
            }
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details or {},
        }


class SSLContextError(CoreError):
    """Raised when SSL context creation fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Unable to create an unverified HTTPS context.",
            ErrorCodes.SSL_CONTEXT_ERROR,
            details=str(details),
        )


class NLTKDownloadError(CoreError):
    """Raised when NLTK corpus download fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Failed to download the 'punkt' corpus from NLTK.",
            ErrorCodes.NLTK_DOWNLOAD_ERROR,
            details=str(details),
        )


class InvalidDatasetInputError(CoreError):
    """Raised when dataset input format is invalid."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid input format for dataset.",
            ErrorCodes.INVALID_DATASET_INPUT,
            details=str(details),
        )


class DatasetLoadError(CoreError):
    """Raised when dataset loading fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Error occurred while loading dataset.",
            ErrorCodes.DATASET_LOAD_ERROR,
            details=str(details),
        )


class InvalidDatasetTypeError(CoreError):
    """Raised when dataset type is invalid."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid dataset type provided.",
            ErrorCodes.INVALID_DATASET_TYPE,
            details=str(details),
        )


class InvalidRegexPatternError(CoreError):
    """Raised when regex pattern compilation fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid regular expression pattern.",
            ErrorCodes.INVALID_REGEX_PATTERN,
            details=str(details),
        )


class DatasetConstraintError(CoreError):
    """Raised when dataset constraint operation fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Error occurred while constraining dataset.",
            ErrorCodes.DATASET_CONSTRAINT_ERROR,
            details=str(details),
        )


class InvalidSentencesInputError(CoreError):
    """Raised when sentences input is invalid."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid sentences input provided.",
            ErrorCodes.INVALID_SENTENCES_INPUT,
            details=str(details),
        )


class InvalidBatchSizeError(CoreError):
    """Raised when batch size is invalid."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid batch size provided.",
            ErrorCodes.INVALID_BATCH_SIZE,
            details=str(details),
        )


class InvalidNumWorkersError(CoreError):
    """Raised when num_workers parameter is invalid."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid num_workers parameter provided.",
            ErrorCodes.INVALID_NUM_WORKERS,
            details=str(details),
        )


class DataLoaderCreationError(CoreError):
    """Raised when DataLoader creation fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Error occurred while creating DataLoader.",
            ErrorCodes.DATALOADER_CREATION_ERROR,
            details=str(details),
        )


class InvalidModelNameError(CoreError):
    """Raised when model name is invalid."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid model name provided.",
            ErrorCodes.INVALID_MODEL_NAME,
            details=str(details),
        )


class ModelCreationError(CoreError):
    """Raised when model creation fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Error occurred while creating model.",
            ErrorCodes.MODEL_CREATION_ERROR,
            details=str(details),
        )


class InvalidModelInputError(CoreError):
    """Raised when model input is invalid."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid model input provided.",
            ErrorCodes.INVALID_MODEL_INPUT,
            details=str(details),
        )


class InvalidTieEncoderDecoderError(CoreError):
    """Raised when tie_encoder_decoder parameter is invalid."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Invalid tie_encoder_decoder parameter provided.",
            ErrorCodes.INVALID_TIE_ENCODER_DECODER,
            details=str(details),
        )


class LossCreationError(CoreError):
    """Raised when loss creation fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Error occurred while creating loss function.",
            ErrorCodes.LOSS_CREATION_ERROR,
            details=str(details),
        )


class TrainingError(CoreError):
    """Raised when training fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Error occurred during training.",
            ErrorCodes.TRAINING_ERROR,
            details=str(details),
        )


class ValidationError(CoreError):
    """Raised when validation fails."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Validation error occurred.",
            ErrorCodes.VALIDATION_ERROR,
            details=str(details),
        )


class TypeMismatchError(CoreError):
    """Raised when type mismatch occurs."""

    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "Type mismatch error occurred.",
            ErrorCodes.TYPE_MISMATCH_ERROR,
            details=str(details),
        )
