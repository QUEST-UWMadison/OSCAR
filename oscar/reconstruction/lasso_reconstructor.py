from typing import Optional

import cvxpy as cp
import numpy as np

from .base_cvxpy_reconstructor import BaseCvxPyReconstructor


class LassoReconstructor(BaseCvxPyReconstructor):
    def __init__(self, tolerance: float, solver: Optional[str] = None, **solver_kwargs):
        self.tolerance = tolerance
        super().__init__(solver, **solver_kwargs)

    def _build_optimization_problem(
        self, A: np.ndarray, x: cp.Variable, b: np.ndarray
    ) -> cp.Problem:
        return cp.Problem(cp.Minimize(cp.norm(A @ x - b, 2)), [cp.norm(x, 1) <= self.tolerance])
