from abc import ABC, abstractmethod

import numpy as np


class BaseExecutor(ABC):
    @abstractmethod
    def run(self, params: np.ndarray) -> float:
        pass

    def run_batch(self, params_list: list[np.ndarray]) -> np.ndarray:
        return np.array([self.run(params) for params in params_list])
