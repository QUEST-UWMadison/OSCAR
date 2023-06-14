from typing import Optional

import cvxpy as cp
import numpy as np

from .base_cvxpy_reconstructor import BaseCvxPyReconstructor


class BPDNVariantReconstructor(BaseCvxPyReconstructor):
    def __init__(self, tolerance: float, solver: Optional[str] = None, **solver_kwargs):
        self.tolerance: float = tolerance
        super().__init__(solver, **solver_kwargs)

    def _build_optimization_problem(
        self, A: np.ndarray, x: cp.Variable, b: np.ndarray
    ) -> cp.Problem:
        return cp.Problem(cp.Minimize(cp.norm(x, 1)), [cp.norm(A @ x - b, 2) <= self.tolerance])
