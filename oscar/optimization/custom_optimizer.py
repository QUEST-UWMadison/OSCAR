from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .base_optimizer import BaseOptimizer
from .trace import Trace

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor


class CustomOptimizer(BaseOptimizer):
    def __init__(self, optimizer: Callable[[NDArray[np.float_]], Mapping[str, Any]]) -> None:
        self.optimizer: Callable[[Sequence[float]], Mapping[str, Any]] = optimizer

    def name(self, include_library_name: bool = True) -> str:
        name = self.optimizer.__name__
        if include_library_name:
            name += " (Custom)"
        return name

    def run(
        self, executor: BaseExecutor, initial_point: Sequence[float], *args, **kwargs
    ) -> tuple[Trace, Mapping[str, Any]]:
        trace = Trace()
        result = self.optimizer(
            partial(executor.run, callback=trace.append), np.array(initial_point), *args, **kwargs
        )
        trace.update_with_custom_result(result)
        return trace, result
