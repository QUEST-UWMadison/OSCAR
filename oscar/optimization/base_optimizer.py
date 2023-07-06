from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from qiskit.algorithms.optimizers import OptimizerResult
from scipy.optimize import OptimizeResult

from ..execution.base_executor import BaseExecutor
from .trace import Trace


class BaseOptimizer(ABC):
    @abstractmethod
    def run(
        self, executor: BaseExecutor, initial_point: Sequence[float], *args, **kwargs
    ) -> tuple[Trace, OptimizerResult | OptimizeResult]:
        pass
