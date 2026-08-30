"""Centralised pipeline constants.

All tunable/magic values are declared here so that no literal numbers
appear scattered throughout the business logic.
"""


class AudioConfig:
    """Audio stream configuration."""

    INPUT_SAMPLE_RATE: int = 48000
    TARGET_SAMPLE_RATE: int = 16000
    CHANNELS: int = 2
    BLOCK_SIZE: int = 960          # 20 ms @ 48 kHz

    # Float -> PCM16 range
    PCM16_SCALE: int = 32767


class PipelineConfig:
    """Concurrency and buffering configuration."""

    MAX_INPUT_QUEUE: int = 100
    NUM_WORKERS: int = 4
    RING_SIZE: int = 200

    # Polling interval (seconds) when a queue is empty.
    CONSUMER_POLL_SECONDS: float = 0.001


class MelConfig:
    """Mel-spectrogram feature extraction configuration.

    Windows of audio are accumulated into fixed-length segments (default 30 s)
    before log-Mel features are extracted for STT ingestion.
    """

    SAMPLE_RATE: int = 16000
    WINDOW_SECONDS: int = 30
    WINDOW_SAMPLES: int = SAMPLE_RATE * WINDOW_SECONDS  # 480,000
    N_FFT: int = 400
    HOP_LENGTH: int = 160
    N_MELS: int = 80
