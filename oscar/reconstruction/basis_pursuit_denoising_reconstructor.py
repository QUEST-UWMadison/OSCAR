from typing import Optional

import cvxpy as cp
import numpy as np

from .base_cvxpy_reconstructor import BaseCvxPyReconstructor


class BPDNReconstructor(BaseCvxPyReconstructor):
    def __init__(
        self,
        normalization_factor: Optional[float] = 0.001,
        solver: Optional[str] = None,
        **solver_kwargs
    ):
        self.normalization_factor = normalization_factor
        super().__init__(solver, **solver_kwargs)

    def _build_optimization_problem(
        self, A: np.ndarray, x: cp.Variable, b: np.ndarray
    ) -> cp.Problem:
        return cp.Problem(
            cp.Minimize(self.normalization_factor * cp.norm(x, 1) + cp.norm(A @ x - b, 2) ** 2)
        )
