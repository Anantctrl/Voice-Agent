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

    # Polling interval (seconds) when the ring queue is empty.
    CONSUMER_POLL_SECONDS: float = 0.001
