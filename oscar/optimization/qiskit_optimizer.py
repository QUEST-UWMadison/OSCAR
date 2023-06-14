from functools import partial
from typing import Callable

import numpy as np
from qiskit.algorithms.optimizers import Optimizer, OptimizerResult

from .base_optimizer import BaseOptimizer
from ..execution.base_executor import BaseExecutor
from .trace import Trace


class QiskitOptimizer(BaseOptimizer):
    def __init__(self, qiskit_optimizer: Optimizer):
        self.optimizer: Optimizer = qiskit_optimizer

    def run(
        self,
        executor: BaseExecutor,
        initial_point: np.ndarray,
        jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> tuple[Trace, OptimizerResult]:
        trace = Trace()
        result = self.optimizer.minimize(
            partial(executor.run, callback=trace.append), initial_point, jacobian, bounds
        )
        trace.update_with_qiskit_result(result)
        return trace, result
