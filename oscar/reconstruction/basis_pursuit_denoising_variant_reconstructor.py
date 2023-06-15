import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .base_cvxpy_reconstructor import BaseCvxPyReconstructor


class BPDNVariantReconstructor(BaseCvxPyReconstructor):
    def __init__(self, tolerance: float, solver: str | None = None, **solver_kwargs) -> None:
        self.tolerance: float = tolerance
        super().__init__(solver, **solver_kwargs)

    def _build_optimization_problem(
        self, A: NDArray[np.float_], x: cp.Variable, b: NDArray[np.float_]
    ) -> cp.Problem:
        return cp.Problem(cp.Minimize(cp.norm(x, 1)), [cp.norm(A @ x - b, 2) <= self.tolerance])
