from abc import ABC, abstractmethod

import numpy as np


class BaseExecutor(ABC):
    @abstractmethod
    def run(self, params: np.ndarray) -> float:
        pass
