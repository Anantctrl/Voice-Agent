"""Worker pool (dispatcher) that processes raw chunks concurrently."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from threading import Condition
from typing import Any

from ..constants.audio import PipelineConfig
from ..models.audio_chunk import ProcessingFailure
from .chunk_processor import ChunkProcessor
from .input_queue import InputQueue
from .ring_queue import CompletedQueue
from .producer import STOP

logger = logging.getLogger(__name__)


class WorkerPool:
    """Consumes the input queue and submits each chunk to a worker thread.

    Each task's result is pushed onto the completed queue via a done-callback,
    mirroring the dispatcher design. The STOP sentinel is only emitted once all
    submitted tasks have finished, guaranteeing the completed queue is an
    ordered stream terminated by STOP (no real chunks lost during shutdown).
    """

    def __init__(
        self,
        input_queue: InputQueue,
        completed_queue: CompletedQueue,
        processor: ChunkProcessor,
        *,
        num_workers: int = PipelineConfig.NUM_WORKERS,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self._input_queue = input_queue
        self._completed_queue = completed_queue
        self._processor = processor
        self._num_workers = num_workers
        self._executor: ThreadPoolExecutor | None = None
        self._in_flight = 0
        self._condition = Condition()
        self._stop_requested = False

    def _process(self, item: Any) -> Any:
        """Worker body; returns the processed chunk or the sentinel."""
        if item is STOP:
            return STOP
        sequence, block = item
        return self._processor.process(sequence, block)

    def _counting_callback(self, future, sequence: int) -> None:
        """Handles a completed worker task, emitting a failure marker on error."""
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - surface worker errors
            logger.warning("Worker error on chunk %s: %s", sequence, exc)
            result = ProcessingFailure(sequence=sequence, error=exc)
        finally:
            with self._condition:
                self._in_flight -= 1
                if self._in_flight == 0 and self._stop_requested:
                    self._condition.notify_all()

        self._completed_queue.put(result)

    def _dispatch(self, item: Any) -> None:
        with self._condition:
            self._in_flight += 1
        sequence, block = item
        future = self._executor.submit(self._process, item)
        future.add_done_callback(partial(self._counting_callback, sequence=sequence))

    def run(self) -> None:
        """Dispatch loop; runs until the STOP sentinel is observed."""
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            self._executor = executor
            while True:
                item = self._input_queue.get()
                self._input_queue.task_done()
                if item is STOP:
                    with self._condition:
                        self._stop_requested = True
                        while self._in_flight > 0:
                            self._condition.wait()
                    self._completed_queue.put(STOP)
                    break
                self._dispatch(item)
