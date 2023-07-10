from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from nlopt import opt
from numpy.typing import NDArray

from .base_optimizer import BaseOptimizer
from .trace import Trace

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor


class NLoptOptimizer(BaseOptimizer):
    def __init__(self, optimizer: opt) -> None:
        self.optimizer: opt = optimizer

    def name(self, include_library_name: bool = True) -> str:
        name = self.optimizer.get_algorithm_name()
        if include_library_name:
            name += " (NLopt)"
        return name

    def run(
        self, executor: BaseExecutor, initial_point: Sequence[float], *args, **kwargs
    ) -> tuple[Trace, opt]:
        trace = Trace()

        def objective_wrapper(params: Sequence[float], gradient: NDArray[np.float_]) -> float:
            if gradient.size > 0:
                raise NotImplementedError("NLopt gradient-based algorithms are not supported yet.")
            return executor.run(params, callback=trace.append)

        self.optimizer.set_min_objective(objective_wrapper)
        optimal_params = self.optimizer.optimize(np.array(initial_point), *args, **kwargs)
        trace.update_with_nlopt_result(self.optimizer, optimal_params)
        return trace, self.optimizer
