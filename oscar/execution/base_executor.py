from abc import ABC, abstractmethod
from copy import deepcopy
from time import time
from typing import Callable, Optional

import numpy as np

from ..optimization import Trace


class BaseExecutor(ABC):
    @abstractmethod
    def _run(self, params: np.ndarray) -> float:
        pass

    def run(
        self,
        params: np.ndarray,
        callback: Optional[Callable[[np.ndarray, float, float], None]] = None,
        return_time: bool = False,
    ) -> float | tuple[float, float]:
        start_time = time()
        value = self._run(params)
        runtime = time() - start_time
        if callback is not None:
            callback(params, value, runtime)
        if return_time:
            return value, time
        return value

    def run_batch(
        self,
        params_list: list[np.ndarray],
        callback: Optional[Callable[[np.ndarray, float, float], None]] = None,
        return_time: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]:
        result = np.array([self.run(params, callback, return_time) for params in params_list])
        if return_time:
            return result.T[0], result.T[1]
        return result

    def run_with_trace(
        self,
        trace: Trace,
        callback: Optional[Callable[[np.ndarray, float, float], None]] = None,
    ) -> Trace:
        new_trace = Trace()
        new_trace.params_trace = deepcopy(trace.params_trace)
        value, runtime = self.run_batch(trace.params_trace, callback, True)
        new_trace.params_value = value.to_list()
        if isinstance(runtime, np.ndarray):
            new_trace.time_trace = runtime.to_list()
        return new_trace
