from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from scipy.optimize import OptimizeResult
from qiskit.algorithms.optimizers import OptimizerResult

from ..execution.base_executor import BaseExecutor
from .trace import Trace


class BaseOptimizer(ABC):
    @abstractmethod
    def run(
        self, executor: BaseExecutor, initial_point: Sequence[float]
    ) -> tuple[Trace, OptimizerResult | OptimizeResult]:
        pass
