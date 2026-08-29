"""PipelineController: wires and coordinates the full processing pipeline.

Top-down coordinator that constructs the queues, processor, worker pool,
reorder buffer and consumer, then runs the producer on the calling thread
while the other stages execute on daemon background threads.

The producer is created via a factory callable so it can receive the input
queue that the controller constructs internally.
"""

from typing import Callable, Optional, Tuple

from ..constants.audio import PipelineConfig
from .chunk_processor import ChunkProcessor, MonoResamplerPcm16Processor
from .consumer import Consumer, NoOpSpeechToTextSink, SpeechToTextSink
from .input_queue import InputQueue
from .producer import Producer
from .reorder_buffer import ReorderBuffer
from .ring_queue import CompletedQueue, RingQueue
from .sequence_assigner import SequenceAssigner
from .worker_pool import WorkerPool

ProducerFactory = Callable[[InputQueue, SequenceAssigner], Producer]
"""Factory signature: build a Producer given the input queue + sequence assigner."""


class PipelineController:
    """Coordinates the producer -> workers -> reorder -> ring -> consumer flow.

    Args:
        producer_factory: Callable ``(input_queue, assigner) -> Producer``.
    """

    def __init__(
        self,
        producer_factory: ProducerFactory,
        *,
        processor: Optional[ChunkProcessor] = None,
        sink: Optional[SpeechToTextSink] = None,
        max_input_queue: int = PipelineConfig.MAX_INPUT_QUEUE,
        num_workers: int = PipelineConfig.NUM_WORKERS,
        ring_size: int = PipelineConfig.RING_SIZE,
    ) -> None:
        self._processor = processor or MonoResamplerPcm16Processor()
        self._sink = sink or NoOpSpeechToTextSink()

        self._input_queue = InputQueue(maxsize=max_input_queue)
        self._assigner = SequenceAssigner()
        self._completed_queue = CompletedQueue()
        self._ring_queue = RingQueue(size=ring_size)

        self._producer = producer_factory(self._input_queue, self._assigner)

        self._worker_pool = WorkerPool(
            self._input_queue,
            self._completed_queue,
            self._processor,
            num_workers=num_workers,
        )
        self._reorder = ReorderBuffer(self._completed_queue, self._ring_queue)
        self._consumer = Consumer(self._ring_queue, self._sink)

    def run(self) -> None:
        """Start background stages and run the producer on the calling thread."""
        import threading

        threads = [
            threading.Thread(target=self._worker_pool.run, name="worker-pool", daemon=True),
            threading.Thread(target=self._reorder.run, name="reorder", daemon=True),
            threading.Thread(target=self._consumer.run, name="consumer", daemon=True),
        ]
        for thread in threads:
            thread.start()

        try:
            self._producer.start()
        finally:
            self._input_queue.join()

        for thread in threads:
            thread.join(timeout=5.0)

    @property
    def input_queue(self) -> InputQueue:
        return self._input_queue

    @property
    def ring_queue(self) -> RingQueue:
        return self._ring_queue

    @property
    def consumer(self) -> Consumer:
        return self._consumer

    @property
    def processor(self) -> ChunkProcessor:
        return self._processor
