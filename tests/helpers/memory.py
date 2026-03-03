import tracemalloc
from types import TracebackType
from typing import Type


class TracemallocTracker:
    _finished = False
    _usage = None

    def __enter__(self):
        tracemalloc.start()
        return self

    @property
    def current(self):
        if not self._finished:
            raise RuntimeError("Context manager has not exited yet")
        return self._usage[0]

    @property
    def peak(self):
        if not self._finished:
            raise RuntimeError("Context manager has not exited yet")
        return self._usage[1]

    def __exit__(
        self,
        exc_type: Type[Exception] | None,
        exc_value: Exception | None,
        exc_traceback: TracebackType | None,
    ) -> bool | None:
        self._finished = True
        self._usage = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        return None


def track_peak_memory_usage():
    return TracemallocTracker()
