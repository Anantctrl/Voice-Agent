"""Standalone Whisper speech-to-text transcription.

Converts an audio file (expected mono 16 kHz PCM16, e.g. the pipeline's
``newenhanced.wav`` output) into text using the cached ``openai-whisper``
``base`` model on CPU.

Usage:
    python -m src.audiopreprocessing.stt.whisper_transcriber --input newenhanced.wav
"""

import argparse
import os
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file to text using Whisper."
    )
    parser.add_argument("--input", required=True, metavar="WAV",
                        help="Audio file to transcribe (e.g. newenhanced.wav).")
    parser.add_argument("--model", default="base", metavar="NAME",
                        help="Whisper model name (default: base, cached).")
    parser.add_argument("--output", default=None, metavar="TXT",
                        help="Transcript file path (default: <input_basename>.txt).")
    parser.add_argument("--language", default="en", metavar="CODE",
                        help="Language code (default: en). Use '' to auto-detect.")
    return parser


def default_output_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return base + ".txt"


def load_audio_array(input_path: str) -> "np.ndarray":
    """Load audio and return float32 mono at 16 kHz (bypasses ffmpeg).

    ``openai-whisper``'s built-in ``load_audio`` shells out to ffmpeg, which is
    not installed here. We decode with soundfile and resample with scipy
    instead, producing the 16 kHz float32 buffer Whisper expects.
    """
    import soundfile as sf
    import numpy as np
    from ..utils.audio_utils import resample

    with sf.SoundFile(input_path) as snd:
        sample_rate = int(snd.samplerate)
        channels = snd.channels
        data = snd.read(dtype="float32", always_2d=True)

    if data.ndim == 2 and data.shape[1] > 1:
        data = np.mean(data, axis=1)
    else:
        data = data[:, 0]

    if sample_rate != 16000:
        data = resample(data, sample_rate, 16000)

    return np.ascontiguousarray(data, dtype=np.float32)


def transcribe_audio(
    input_path: str,
    *,
    model_name: str = "base",
    language: Optional[str] = "en",
) -> str:
    """Transcribe an audio file and return the full transcript text."""
    import whisper

    model = whisper.load_model(model_name)
    audio = load_audio_array(input_path)

    kwargs = {}
    if language:
        kwargs["language"] = language

    result = model.transcribe(audio, **kwargs)

    segments: List[str] = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if text:
            segments.append(text)

    return " ".join(segments).strip() if segments else ""


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    language = args.language if args.language else None
    output_path = args.output or default_output_path(args.input)

    print(f"Transcribing {args.input} with whisper model '{args.model}' ...")
    transcript = transcribe_audio(
        args.input, model_name=args.model, language=language
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(transcript + "\n")

    print(f"\nTranscript ({output_path}):")
    print(transcript)
    print(f"\nSaved transcript to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
