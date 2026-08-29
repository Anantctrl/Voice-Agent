"""Exception hierarchy for the audio preprocessing pipeline."""


class AudioProcessingError(Exception):
    """Base error for all pipeline failures."""

    def __init__(self, message: str, *, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause


class QueueFullError(AudioProcessingError):
    """Raised when pushing to a bounded queue that is currently full."""


class InvalidAudioInputError(AudioProcessingError, ValueError):
    """Raised when audio data fails validation (shape, dtype, sample rate)."""


class PipelineStoppedError(AudioProcessingError):
    """Raised when a pipeline component is used after shutdown."""
