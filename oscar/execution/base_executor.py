from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import deepcopy
from time import time

import numpy as np
from numpy.typing import NDArray

from ..optimization import Trace


class BaseExecutor(ABC):
    @abstractmethod
    def _run(self, params: Sequence[float], **kwargs) -> float:
        pass

    def run(
        self,
        params: Sequence[float],
        callback: Callable[[Sequence[float], float, float], None] | None = None,
        return_time: bool = False,
        **kwargs,
    ) -> float | tuple[float, float]:
        start_time = time()
        value = self._run(params, **kwargs)
        runtime = time() - start_time
        if callback is not None:
            callback(params, value, runtime)
        if return_time:
            return value, runtime
        return value

    def run_batch(
        self,
        params_list: Sequence[Sequence[float]],
        callback: Callable[[Sequence[float], float, float], None] | None = None,
        return_time: bool = False,
        **kwargs,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        result = np.array(
            [self.run(params, callback, return_time, **kwargs) for params in params_list]
        )
        if return_time:
            return result.T[0], result.T[1]
        return result

    def run_with_trace(
        self,
        trace: Trace,
        callback: Callable[[Sequence[float], float, float], None] | None = None,
        **kwargs,
    ) -> Trace:
        new_trace = Trace()
        new_trace.params_trace = deepcopy(trace.params_trace)
        value, runtime = self.run_batch(trace.params_trace, callback, True, **kwargs)
        new_trace.value_trace = value.tolist()
        if isinstance(runtime, np.ndarray):
            new_trace.time_trace = runtime.tolist()
        return new_trace
