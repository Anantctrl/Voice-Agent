# AUDIO PREPROCESSING PIPELINE — VISUAL ARCHITECTURE

## Approach (ASCII diagram)

        PRODUCER
  (FileProducer / MicrophoneProducer)
      48 kHz, BLOCK_SIZE = 960 (20 ms)
            │
            ▼
       Sequence Assigner
       (0, 1, 2, 3, ...)
            │
            ▼
        InputQueue               ────────────────────────────
            │                                                    │
    ┌───────┼───────────┬───────────┬───────────┐        (thread-safe)
    ▼       ▼           ▼           ▼           ▼                │
 Worker 1  Worker 2   Worker 3   Worker 4   ... Worker N         │
    │  (NUM_WORKERS = 4)              MonoResamplerPcm16Processor │
    │  stereo→mono → resample 48k→16k → PCM16 (int16)             │
    ▼       ▼           ▼           ▼           ▼                │
    └───────┼───────────┴───────────┴───────────┘                │
            ▼                                                     │
      CompletedQueue                                             │
            │                                                     │
            ▼                                                     │
      Reorder Buffer                                              │
   (wait for next expected index)                                │
        skips ProcessingFailure / failed_count                   │
            │                                                     │
            0 → 1 → 2 → 3 → 4 ...                                │
            ▼                                                     │
        Ring Queue                                               │
   (ordered 16 kHz PCM16 chunks, RING_SIZE=200)      ────────────┘
            │
            ▼
   ┌───────────────────────────────────────────────┐
   │        WindowAccumulator                      │
   │   buffers to WINDOW_SAMPLES = 480,000 (30 s)  │────(optional)──▶ Pcm16WavWriter ──▶ output_wav
   │   overshoot carried to next window            │
   └───────────────────────────────────────────────┘
            │
            ▼
       window_queue (CompletedQueue)
            │
            ▼            
   ┌───────────────────────────────────────────────┐
   │        Consumer                               │
   │   MelSpectrogramExtractor.extract(window)     │
   │        → (80, ~3000) log-Mel                  │
   └───────────────────────────────────────────────┘
            │
            ▼
      SpeechToTextSink.feed(mel)
            │
            ▼
        Whisper STT
   (python -m ...stt.whisper_transcriber)
       base / small model

----------------------------------------------------------------------------------------------
## Full pipeline in one flow

 Producer ─▶ InputQueue ─▶ WorkerPool(N=4) ─▶ CompletedQueue ─▶ ReorderBuffer ─▶ RingQueue
    ─▶ WindowAccumulator ─▶ window_queue ─▶ Consumer(MelSpectrogramExtractor) ─▶ Sink ─▶ Whisper
        │                                     │
        └──(optional)──▶ newenhanced.wav      └──▶ newenhanced.txt

## Offline end-to-end usage (48 kHz WAV → enhanced WAV → transcript)

    python -m src.audiopreprocessing.main \
        --input my_voice_raw.wav --output newenhanced.wav

    python -m src.audiopreprocessing.stt.whisper_transcriber \
        --input newenhanced.wav --model small --output newenhanced.txt
