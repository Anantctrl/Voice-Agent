"""Application entry point for the audio preprocessing pipeline (VAD front-end).

Usage:
    python -m src.audiopreprocessing.main
        -> live microphone capture through the full pipeline.

    python -m src.audiopreprocessing.main --input <file.wav>
        -> offline replay of a WAV through the exact same pipeline.
"""

import argparse
from typing import List, Optional

from .constants.audio import AudioConfig
from .models.audio_chunk import AudioChunk
from .services.consumer import SpeechToTextSink
from .services.pipeline_controller import PipelineController
from .services.producer import FileProducer, MicrophoneProducer


class LoggingSink(SpeechToTextSink):
    """Diagnostic sink: logs each ordered chunk delivered to the STT stage."""

    def feed(self, chunk: AudioChunk) -> bool:
        print(
            f"Sending chunk {chunk.sequence} to STT "
            f"({chunk.num_samples} samples @ {chunk.output_sample_rate} Hz)"
        )
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audio preprocessing pipeline for VAD "
                    "(mono -> 16 kHz -> PCM16)."
    )
    parser.add_argument(
        "--input",
        metavar="WAV",
        default=None,
        help="Replay a WAV file offline instead of capturing from the microphone.",
    )
    return parser


def build_producer_factory(args: argparse.Namespace):
    """Return a producer factory bound to the chosen input mode."""
    if args.input:
        def from_file(queue, assigner):
            return FileProducer(
                queue, assigner, args.input, block_size=AudioConfig.BLOCK_SIZE
            )
        return from_file

    def from_mic(queue, assigner):
        return MicrophoneProducer(
            queue, assigner,
            sample_rate=AudioConfig.INPUT_SAMPLE_RATE,
            channels=AudioConfig.CHANNELS,
            block_size=AudioConfig.BLOCK_SIZE,
        )
    return from_mic


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.input:
        print(f"Offline mode: replaying {args.input}")
    else:
        print("Live mode: capturing from microphone (Ctrl+C to stop).")

    controller = PipelineController(
        build_producer_factory(args),
        sink=LoggingSink(),
    )

    controller.run()
    print(f"Done. Chunks consumed by STT stage: {controller.consumer.consumed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
