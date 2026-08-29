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



