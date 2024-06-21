from __future__ import annotations

import warnings
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qiskit.algorithms import optimizers
from qiskit.algorithms.optimizers import Optimizer

from .base_optimizer import BaseOptimizer, ConstraintsType, JacobianType

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor, CallbackType


class QiskitOptimizer(BaseOptimizer):
    def __init__(self, optimizer: str | Optimizer, **configs) -> None:
        if isinstance(optimizer, str):
            self.optimizer: Optimizer = getattr(optimizers, optimizer)()
        else:
            self.optimizer: Optimizer = optimizer
        self.optimizer.set_options(**configs)
        super().__init__()

    def name(self, include_library_name: bool = True) -> str:
        name = self.optimizer.__class__.__name__
        if include_library_name:
            name += " (Qiskit)"
        return name

    def _run(
        self,
        executor: BaseExecutor,
        initial_point: NDArray[np.float64],
        bounds: NDArray[np.float64] | None = None,
        jacobian: JacobianType | None = None,
        constraints: ConstraintsType | None = None,
        callback: CallbackType | None = None,
        **executor_kwargs,
    ) -> None:
        if constraints is not None:
            warnings.warn("Constraints are ignored for Qiskit methods.")
        self.result = self.optimizer.minimize(
            self._objective(executor, callback, **executor_kwargs),
            initial_point,
            jacobian,
            bounds,
        )
