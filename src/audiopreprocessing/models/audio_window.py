"""Data model for a fixed-length window of audio used for feature extraction."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AudioWindow:
    """A fixed-length window of normalized audio ready for feature extraction.

    Attributes:
        start_sequence: Sequence of the first chunk folded into this window.
        end_sequence: Sequence of the last chunk folded into this window.
        samples: Mono float32 samples normalized to [-1, 1], length equal to
            the configured window size (or zero-padded to that length).
        is_padded: True when the final (short) window was zero-padded to reach
            the full window length on shutdown.
    """

    start_sequence: int
    end_sequence: int
    samples: np.ndarray
    is_padded: bool

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a numpy ndarray")
        if self.samples.dtype != np.float32:
            raise TypeError("samples must be float32")
        if self.samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono buffer")
