from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from numpy.typing import NDArray
from qiskit.algorithms.optimizers import Optimizer, OptimizerResult

from ..execution.base_executor import BaseExecutor
from .base_optimizer import BaseOptimizer
from .trace import Trace


class QiskitOptimizer(BaseOptimizer):
    def __init__(self, qiskit_optimizer: Optimizer) -> None:
        self.optimizer: Optimizer = qiskit_optimizer

    def run(
        self,
        executor: BaseExecutor,
        initial_point: Sequence[float],
        jacobian: Callable[[NDArray[np.float_]], NDArray[np.float_]] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> tuple[Trace, OptimizerResult]:
        trace = Trace()
        result = self.optimizer.minimize(
            partial(executor.run, callback=trace.append), initial_point, jacobian, bounds
        )
        trace.update_with_qiskit_result(result)
        return trace, result
