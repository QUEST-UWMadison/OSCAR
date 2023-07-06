from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from nlopt import opt
from numpy.typing import NDArray

from ..execution.base_executor import BaseExecutor
from .base_optimizer import BaseOptimizer
from .trace import Trace


class NLoptOptimizer(BaseOptimizer):
    def __init__(self, optimizer: opt) -> None:
        self.optimizer: opt = optimizer

    def name(self, include_library_name: bool = True) -> str:
        name = self.optimizer.get_algorithm_name()
        if include_library_name:
            name += " (NLopt)"
        return name

    def run(
        self,
        executor: BaseExecutor,
        initial_point: Sequence[float],
    ) -> tuple[Trace, opt]:
        trace = Trace()
        self.optimizer.set_min_objective(partial(executor.run, callback=trace.append))
        self.optimizer.optimize(np.array(initial_point))
        trace.update_with_nlopt_result(self.optimizer)
        return trace, self.optimizer
