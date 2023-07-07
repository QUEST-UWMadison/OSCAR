from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from qiskit.algorithms.optimizers import OptimizerResult
from scipy.optimize import OptimizeResult

from .trace import Trace

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor


class BaseOptimizer(ABC):
    @abstractmethod
    def name(self, include_library_name: bool = True) -> str:
        pass

    @abstractmethod
    def run(
        self, executor: BaseExecutor, initial_point: Sequence[float], *args, **kwargs
    ) -> tuple[Trace, OptimizerResult | OptimizeResult]:
        pass
