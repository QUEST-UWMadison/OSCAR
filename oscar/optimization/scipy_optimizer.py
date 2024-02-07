from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .base_optimizer import BaseOptimizer, ConstraintsType, JacobianType

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor, CallbackType


class SciPyOptimizer(BaseOptimizer):
    def __init__(
        self, optimizer: str, tol: float | None = None, options: dict[str, Any] | None = None
    ) -> None:
        self.optimizer: str = optimizer
        self.tol: float | None = tol
        self.options: dict[str, Any] | None = options
        super().__init__()

    def name(self, include_library_name: bool = True) -> str:
        return self.optimizer + (" (SciPy)" if include_library_name else "")

    def _run(
        self,
        executor: BaseExecutor,
        initial_point: NDArray[np.float_],
        bounds: NDArray[np.float_] | None = None,
        jacobian: JacobianType | None = None,
        constraints: ConstraintsType | None = None,
        callback: CallbackType | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.result = minimize(
            self._objective(executor, callback, **executor_kwargs),
            x0=initial_point,
            method=self.optimizer,
            jac=jacobian,
            bounds=bounds,
            constraints=to_scipy_constraints(constraints),
            tol=self.tol,
            options=self.options,
            **kwargs,
        )


def to_scipy_constraints(constraints: ConstraintsType | None) -> list[dict[str, Any]] | None:
    if constraints is None:
        return None
    constraints = list(constraints)
    for i, constraint in enumerate(constraints):
        if len(constraint) == 2:
            constraint += (None,)
        constraints[i] = {
            "type": constraint[0],
            "fun": constraint[1],
            "jac": constraint[2],
        }
    return constraints
