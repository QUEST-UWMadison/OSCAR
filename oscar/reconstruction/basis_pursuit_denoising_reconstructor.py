from __future__ import annotations

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .base_cvxpy_reconstructor import BaseCvxPyReconstructor


class BPDNReconstructor(BaseCvxPyReconstructor):
    def __init__(
        self, normalization_factor: float = 0.001, solver: str | None = None, **solver_kwargs
    ) -> None:
        self.normalization_factor: float = normalization_factor
        super().__init__(solver, **solver_kwargs)

    def _build_optimization_problem(
        self, A: NDArray[np.float64], x: cp.Variable, b: NDArray[np.float64]
    ) -> cp.Problem:
        return cp.Problem(
            cp.Minimize(self.normalization_factor * cp.norm(x, 1) + cp.norm(A @ x - b, 2) ** 2)
        )
