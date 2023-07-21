from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial, singledispatchmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qiskit.algorithms import optimizers
from qiskit.algorithms.optimizers import Optimizer, OptimizerResult

from .base_optimizer import BaseOptimizer
from .trace import Trace

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor


class QiskitOptimizer(BaseOptimizer):
    @singledispatchmethod
    def __init__(self, optimizer: Optimizer, **configs) -> None:
        optimizer.set_options(**configs)
        self.optimizer: Optimizer = optimizer

    @__init__.register
    def _(self, optimizer: str, **configs) -> None:
        self.optimizer = getattr(optimizers, optimizer)(**configs)

    def name(self, include_library_name: bool = True) -> str:
        name = self.optimizer.__class__.__name__
        if include_library_name:
            name += " (Qiskit)"
        return name

    def run(
        self,
        executor: BaseExecutor,
        initial_point: Sequence[float],
        executor_kwargs: dict[str, Any] | None = None,
        jacobian: Callable[[NDArray[np.float_]], NDArray[np.float_]] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        **kwargs,
    ) -> tuple[Trace, OptimizerResult]:
        if executor_kwargs is None:
            executor_kwargs = {}
        trace = Trace()
        self.optimizer.set_options(**kwargs)
        result = self.optimizer.minimize(
            partial(executor.run, callback=trace.append, **executor_kwargs),
            np.array(initial_point),
            jacobian,
            bounds,
        )
        trace.update_with_qiskit_result(result)
        return trace, result
