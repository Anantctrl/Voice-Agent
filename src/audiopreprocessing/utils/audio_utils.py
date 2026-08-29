"""Low-level audio manipulation helpers.

These are pure functions with a single responsibility each; they contain no
concurrency or pipeline concerns.
"""

import numpy as np
from scipy.signal import resample_poly

from ..constants.audio import AudioConfig
from ..exceptions.pipeline_errors import InvalidAudioInputError


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Down-mix a (frames, channels) buffer to a single mono channel.

    Args:
        samples: Input audio with shape (N,) for mono or (N, C) for multi.

    Returns:
        Mono float32 samples with shape (N,).

    Raises:
        InvalidAudioInputError: If the input is empty or not 1-D/2-D.
    """
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)

    if samples.ndim == 2:
        if samples.shape[1] < 1:
            raise InvalidAudioInputError("stereo buffer has no channels")
        if samples.shape[0] < 1:
            raise InvalidAudioInputError("audio buffer has no frames")
        return np.mean(samples, axis=1).astype(np.float32)

    raise InvalidAudioInputError(
        f"expected 1-D or 2-D audio buffer, got ndim={samples.ndim}"
    )


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resample mono samples from ``source_rate`` to ``target_rate``.

    Args:
        samples: Mono float32 samples.
        source_rate: Original sample rate in Hz.
        target_rate: Desired sample rate in Hz.

    Returns:
        Resampled mono float32 samples.
    """
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    return resample_poly(samples, target_rate, source_rate).astype(np.float32)


def to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert float samples in [-1, 1] to 16-bit signed PCM.

    Args:
        samples: Float mono samples (assumed already bounded to [-1, 1]).

    Returns:
        int16 numpy array of the same shape.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * AudioConfig.PCM16_SCALE).astype(np.int16)
