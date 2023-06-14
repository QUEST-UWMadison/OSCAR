from abc import ABC, abstractmethod
from time import time
from typing import Callable, Optional

import numpy as np


class BaseExecutor(ABC):
    @abstractmethod
    def _run(self, params: np.ndarray) -> float:
        pass

    def run(
        self,
        params: np.ndarray,
        callback: Optional[Callable[[np.ndarray, float, float], None]] = None,
    ) -> float:
        start_time = time()
        value = self._run(params)
        if callback is not None:
            callback(params, value, time() - start_time)
        return value

    def run_batch(
        self,
        params_list: list[np.ndarray],
        callback: Optional[Callable[[np.ndarray, float, float], None]] = None,
    ) -> np.ndarray:
        return np.array([self.run(params, callback) for params in params_list])
