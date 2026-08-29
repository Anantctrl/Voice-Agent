"""Audio preprocessing pipeline for VAD.

Performs live (or offline) audio capture, converts to mono, resamples to
16 kHz and quantises to 16-bit PCM through a threaded producer/worker/
reorder-buffer/ring-queue architecture.
"""

__version__ = "0.1.0"
