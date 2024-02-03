from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Sequence
from copy import deepcopy
from time import time

from ..optimization import Trace


class BaseExecutor(ABC):
    @abstractmethod
    def _run(self, params: Sequence[float], **kwargs) -> float:
        pass

    def run(
        self,
        params: Sequence[float],
        callback: Callable[[Sequence[float], float, float], None] | None = None,
        **kwargs,
    ) -> float:
        start_time = time()
        value = self._run(params, **kwargs)
        runtime = time() - start_time
        if callback is not None:
            callback(params, value, runtime)
        return value

    def run_batch(
        self,
        params_list: Iterable[Sequence[float]],
        callback: Callable[[Sequence[float], float, float], None] | None = None,
        **kwargs,
    ) -> Generator[float, None, None]:
        for params in params_list:
            yield self.run(params, callback, **kwargs)

    def run_with_trace(
        self,
        trace: Trace,
        callback: Callable[[Sequence[float], float, float], None] | None = None,
        **kwargs,
    ) -> Trace:
        new_trace = Trace()
        new_trace.params_trace = deepcopy(trace.params_trace)
        time_trace = []

        def append_time(params: Sequence[float], value: float, runtime: float) -> None:
            time_trace.append(runtime)
            if callback is not None:
                callback(params, value, runtime)

        new_trace.value_trace = list(self.run_batch(trace.params_trace, append_time, **kwargs))
        new_trace.time_trace = time_trace
        return new_trace
