from .base_executor import BaseExecutor
from typing import Callable
import numpy as np


class CustomExecutor(BaseExecutor):
    def __init__(self, function: Callable[[np.ndarray], float]):
        self.function: Callable[[np.ndarray], float] = function

    def _run(self, params: np.ndarray) -> float:
        return self.function(params)
