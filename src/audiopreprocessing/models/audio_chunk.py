"""Data models representing audio travelling through the pipeline."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """A single processed audio chunk carrying its ordering metadata.

    Attributes:
        sequence: Global monotonic sequence number assigned at capture time.
        samples: Numpy array of mono PCM16 samples (int16).
        input_sample_rate: Sample rate prior to processing.
        output_sample_rate: Sample rate after resampling.
        source_block: The raw (pre-processing) samples, retained for
            diagnostics/debugging when populated.
    """

    sequence: int
    samples: np.ndarray
    input_sample_rate: int
    output_sample_rate: int
    source_block: np.ndarray | None = field(default=None)

    @property
    def num_samples(self) -> int:
        """Number of mono samples in this chunk."""
        return int(self.samples.size)

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a numpy ndarray")
        if self.samples.dtype != np.int16:
            raise TypeError("samples must be PCM16 (int16) after processing")
        if self.samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono buffer")
