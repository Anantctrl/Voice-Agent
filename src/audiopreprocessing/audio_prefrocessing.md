#
# WRITE A CODE IN A FORMAT 

# Professional OOP Coding Style Prompt

Act as a senior software engineer with 15+ years of experience writing production-quality software.

Generate code using **professional software engineering practices**, not beginner or competitive programming style.

## Coding Requirements

* Use **Object-Oriented Programming (OOP)** as the primary design paradigm.
* Follow a **top-down design approach**:

  * Start with a clear `Main` class (or entry point).
  * The `Main` class should only coordinate the program flow.
  * Delegate all work to specialized classes.
* Design the code as if it will become part of a large, maintainable application.
* Follow the **Single Responsibility Principle (SRP)**.
* Keep classes focused on one responsibility.
* Avoid writing all logic inside the main method.
* Separate business logic, data models, utilities, configuration, validation, and services into different classes.
* Use encapsulation properly with private fields and public methods.
* Prefer composition over inheritance unless inheritance is clearly justified.
* Use interfaces and abstract classes where appropriate.
* Follow SOLID principles whenever applicable.
* Make the project modular and easily extensible.

## Code Organization

Organize the code into logical components such as:

* Main (Application Entry)
* Controller / Manager
* Service
* Repository / Data Layer
* Model / Entity
* Utility
* Configuration
* Exception Classes
* Constants
* Helper Classes

If the language supports packages or namespaces, organize them appropriately.

## Coding Style

* Use meaningful class names.
* Use descriptive variable names.
* Use descriptive method names.
* Keep methods short and focused.
* One responsibility per method.
* Avoid duplicate code.
* Use constants instead of magic numbers.
* Use dependency injection where appropriate.
* Add proper error handling.
* Validate inputs.
* Write readable, self-documenting code.
* Add comments only where they improve understanding, not for obvious statements.

## Code Quality

Produce code that is:

* Production-ready
* Modular
* Reusable
* Maintainable
* Scalable
* Testable
* Extensible
* Cleanly formatted
* Easy to debug

Avoid:

* God classes
* Long methods
* Deep nesting
* Global variables
* Duplicate logic
* Hard-coded values
* Poor naming
* Tight coupling

## Output Requirements

For every solution:

1. Explain the architecture briefly.
2. Show the folder/project structure.
3. Generate each class separately.
4. Explain the responsibility of each class.
5. Use modern language features and best practices.
6. Ensure the code compiles and runs.
7. If the project grows later, the architecture should support adding new features with minimal modifications.

Write code exactly as a professional software engineer would for a real-world production application.






## TARGET :

WRITE A CODE FOR AUDIO PREPROCESSING  TASK  FOR VAD (VOICE ACTIVE DETECTION ) .


audio buffer 
covert to mono 
Resample it inot 16k htz

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------------------------
audio buffering  + MONO + resampling + 16pcm


THE APPROACH : 
                  Producer
             (sounddevice callback)
                       │
             Assign Sequence Number
                 (0,1,2,3,...)
                       │
                       ▼
            Thread-Safe Input Queue
                       │
      ┌────────────────┼────────────────┐
      │                │                │
      ▼                ▼                ▼
   Worker 1         Worker 2         Worker 3      ... Worker N
      │                │                │
      │ Process         │ Process        │ Process
      ▼                ▼                ▼
      └──────────────┬──────────────────┘
                     ▼
              Completion Queue
                     │
                     ▼
              Reorder Buffer
        (wait for next expected index)
                     │
         0 → 1 → 2 → 3 → 4 ...
                     ▼
                    Ring Queue
                    
                    │
                    ▼
         
                    

import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import resample_poly
import sounddevice as sd

# =====================================================
# Configuration
# =====================================================

INPUT_SR = 48000
TARGET_SR = 16000
CHANNELS = 2
BLOCKSIZE = 960          # 20 ms @ 48 kHz

MAX_QUEUE = 100
NUM_WORKERS = 4

# Final Ring Buffer
RING_SIZE = 200

# =====================================================
# Queues
# =====================================================

input_queue = queue.Queue(MAX_QUEUE)
completed_queue = queue.Queue()

# =====================================================
# Ring Queue (Final Buffer)
# =====================================================

ring_queue = deque(maxlen=RING_SIZE)
ring_lock = threading.Lock()

# =====================================================
# Sequence Number
# =====================================================

sequence = 0
sequence_lock = threading.Lock()

# =====================================================
# Producer (Microphone Callback)
# =====================================================

def audio_callback(indata, frames, time_info, status):

    global sequence

    if status:
        print(status)

    with sequence_lock:
        idx = sequence
        sequence += 1

    try:
        input_queue.put_nowait((idx, indata.copy()))

    except queue.Full:
        print("Input queue full - dropping frame")


# =====================================================
# Audio Processing
# =====================================================

def process_chunk(index, audio):

    # Stereo -> Mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample 48k -> 16k
    audio = resample_poly(audio, TARGET_SR, INPUT_SR)

    # Float -> PCM16
    audio = np.clip(audio, -1, 1)
    audio = (audio * 32767).astype(np.int16)

    return index, audio


# =====================================================
# Dispatcher
# =====================================================

def dispatcher():

    while True:

        index, chunk = input_queue.get()

        future = executor.submit(process_chunk, index, chunk)

        future.add_done_callback(
            lambda f: completed_queue.put(f.result())
        )

        input_queue.task_done()


# =====================================================
# Ordered Delivery
# Stores ordered chunks into ring queue
# =====================================================

def ordered_delivery():

    expected = 0
    reorder_buffer = {}

    while True:

        index, processed = completed_queue.get()

        reorder_buffer[index] = processed

        while expected in reorder_buffer:

            chunk = reorder_buffer.pop(expected)

            with ring_lock:
                if len(ring_queue) == RING_SIZE:
                    print("Ring queue full - oldest chunk discarded")

                ring_queue.append((expected, chunk))

            print(f"Stored chunk {expected} in ring queue")

            expected += 1

        completed_queue.task_done()


# =====================================================
# Consumer
# Reads from ring queue
# Replace print() with stt.feed(chunk)
# =====================================================

def stt_sender():

    while True:

        with ring_lock:

            if ring_queue:
                index, chunk = ring_queue.popleft()
            else:
                index = None

        if index is None:
            threading.Event().wait(0.001)
            continue

        print(f"Sending chunk {index} to STT")

        # stt.feed(chunk)


# =====================================================
# Thread Pool
# =====================================================

executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

# =====================================================
# Threads
# =====================================================

threading.Thread(
    target=dispatcher,
    daemon=True
).start()

threading.Thread(
    target=ordered_delivery,
    daemon=True
).start()

threading.Thread(
    target=stt_sender,
    daemon=True
).start()

# =====================================================
# Microphone
# =====================================================

with sd.InputStream(
    samplerate=INPUT_SR,
    channels=CHANNELS,
    dtype="float32",
    blocksize=BLOCKSIZE,
    callback=audio_callback,
):

    print("Listening...")

    threading.Event().wait()



# =============================================================================
# MEL-SPECTROGRAM FEATURE-EXTRACTION STAGE
# (Window accumulator + log-Mel extractor, feeding the STT consumer)
# =============================================================================

Now that ordered mono 16 kHz PCM16 chunks reach the ring queue, a new stage
buffers them into fixed-length windows and converts each window to an 80-bin
log-Mel spectrogram before handing it to the STT backend.

New pipeline layout:

    ReorderBuffer → RingQueue → WindowAccumulator
                                    │
                               window_queue
                                    ▼
                        MelSpectrogramExtractor (FeatureExtractor)
                                    │
                               consumer → sink.feed(mel)
                                    │
                                   STT

## Configuration (constants/audio.py -> MelConfig)

    class MelConfig:
        SAMPLE_RATE: int = 16000
        WINDOW_SECONDS: int = 30
        WINDOW_SAMPLES: int = SAMPLE_RATE * WINDOW_SECONDS   # 480,000
        N_FFT: int = 400
        HOP_LENGTH: int = 160
        N_MELS: int = 80

## AudioWindow model (models/audio_window.py)

    @dataclass(frozen=True)
    class AudioWindow:
        start_sequence: int      # first chunk folded into this window
        end_sequence: int        # last chunk folded into this window
        samples: np.ndarray      # float32 normalized [-1,1], length == WINDOW_SAMPLES
        is_padded: bool          # True if zero-padded to reach a full window

    __post_init__ validates dtype == float32 and ndim == 1.

## WindowAccumulator (services/window_accumulator.py)

Reads ordered PCM16 chunks from the ring queue, converts each to float32 via
`chunk.samples / AudioConfig.PCM16_SCALE` (32767), and buffers them until at
least `WINDOW_SAMPLES` (480,000) samples are present. On full: `_flush(pad=False)`
emits an `AudioWindow` and resets the buffer. On the producer's STOP sentinel:
if a partial buffer remains it is `_flush(pad=True)` zero-padded, then STOP is
forwarded to the window queue.

  * `_flush(pad)` concatenates buffered chunks, zero-pads a short window when
    `pad` is True, truncates any overshoot beyond `WINDOW_SAMPLES`, then enqueues.
  * Empty ring-queue pops back off for `poll_seconds` (from
    `PipelineConfig.CONSUMER_POLL_SECONDS`, 1 ms) instead of busy-spinning —
    the same throttled-polling pattern used by the Consumer stage.

## FeatureExtractor / MelSpectrogramExtractor (services/feature_extractor.py)

    class FeatureExtractor(ABC):
        @abstractmethod
        def extract(self, window: AudioWindow) -> np.ndarray: ...

    class MelSpectrogramExtractor(FeatureExtractor):
        def __init__(self, *, n_fft=MelConfig.N_FFT,
                     hop_length=MelConfig.HOP_LENGTH, n_mels=MelConfig.N_MELS,
                     sample_rate=MelConfig.SAMPLE_RATE):
            # note: hop_length is NOT passed to filters.mel
            self._mel_filters = librosa.filters.mel(sr=sample_rate,
                                                    n_fft=n_fft, n_mels=n_mels)

        def extract(self, window):
            stft = librosa.stft(window.samples, n_fft=self._n_fft,
                                hop_length=self._hop_length, window="hann")
            power = np.abs(stft) ** 2
            mel = self._mel_filters @ power
            log_mel = np.log10(np.clip(mel, a_min=1e-10, a_max=None))
            log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
            return ((log_mel + 4.0) / 4.0).astype(np.float32)  # shape (80, ~3000)

Notes:
  * `librosa.filters.mel` (v0.11.0) does NOT accept `hop_length`; it is only
    applied to `librosa.stft`.
  * The final normalization matches Whisper's preprocessing convention; drop it
    if the STT model expects raw log-Mel instead.

## Consumer changes (services/consumer.py)

`SpeechToTextSink.feed()` now accepts a mel feature array instead of an
`AudioChunk`:

    class SpeechToTextSink(ABC):
        @abstractmethod
        def feed(self, mel: np.ndarray) -> bool: ...

The `Consumer` reads `AudioWindow`s from the window queue, runs
`extractor.extract(window)`, and forwards the resulting `(80, ~3000)` array to
`feed()`. `Consumer.consumed_count` now counts mel windows (one per 30 s
window) rather than raw 20 ms chunks.

## Wiring (services/pipeline_controller.py)

`PipelineController` now constructs `window_queue`, `WindowAccumulator`, and
`MelSpectrogramExtractor`, and starts an additional `window-accumulator`
daemon thread between the reorder buffer and the consumer:

    self._window_queue = CompletedQueue()
    self._window_accumulator = WindowAccumulator(self._ring_queue, self._window_queue)
    self._consumer = Consumer(self._window_queue, self._sink, extractor=...)

## Latency tradeoff

This is an intentional architectural batching decision: a 30-second window must
fill before the first feature array is emitted, so downstream STT now waits up
to 30 s per window (or until input end, when a short window is zero-padded and
flushed). The pipeline remains low-latency *up to* the window boundary; a later
optimization could emit partial/sliding windows if that latency proves
unacceptable.





