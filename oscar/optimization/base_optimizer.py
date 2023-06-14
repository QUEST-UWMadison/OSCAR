from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import OptimizeResult
from qiskit.algorithms.optimizers import OptimizerResult

from ..execution.base_executor import BaseExecutor
from .trace import Trace


class BaseOptimizer(ABC):
    @abstractmethod
    def run(
        self, executor: BaseExecutor, initial_point: np.ndarray
    ) -> tuple[Trace, OptimizerResult | OptimizeResult]:
        pass
