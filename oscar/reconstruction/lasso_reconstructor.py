from __future__ import annotations

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .base_cvxpy_reconstructor import BaseCvxPyReconstructor


class LassoReconstructor(BaseCvxPyReconstructor):
    def __init__(self, tolerance: float, solver: str | None = None, **solver_kwargs) -> None:
        self.tolerance: float = tolerance
        super().__init__(solver, **solver_kwargs)

    def _build_optimization_problem(
        self, A: NDArray[np.float_], x: cp.Variable, b: NDArray[np.float_]
    ) -> cp.Problem:
        return cp.Problem(cp.Minimize(cp.norm(A @ x - b, 2)), [cp.norm(x, 1) <= self.tolerance])
