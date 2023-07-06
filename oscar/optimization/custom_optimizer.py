from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..execution.base_executor import BaseExecutor
from .base_optimizer import BaseOptimizer
from .trace import Trace


class CustomOptimizer(BaseOptimizer):
    def __init__(self, optimizer: Callable[[NDArray[np.float_]], Mapping[str, Any]]) -> None:
        self.optimizer: Callable[[Sequence[float]], Mapping[str, Any]] = optimizer

    def run(
        self, executor: BaseExecutor, initial_point: Sequence[float], *args, **kwargs
    ) -> tuple[Trace, Mapping[str, Any]]:
        trace = Trace()
        result = self.optimizer(
            partial(executor.run, callback=trace.append), np.array(initial_point), *args, **kwargs
        )
        trace.update_with_custom_result(result)
        return trace, result
