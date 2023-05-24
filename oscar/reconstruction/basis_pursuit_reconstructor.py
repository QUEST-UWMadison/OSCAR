import cvxpy as cp
import numpy as np

from .base_cvxpy_reconstructor import BaseCvxPyReconstructor


class BPReconstructor(BaseCvxPyReconstructor):
    def _build_optimization_problem(
        self, A: np.ndarray, x: cp.Variable, b: np.ndarray
    ) -> cp.Problem:
        return cp.Problem(cp.Minimize(cp.norm(x, 1)), [A @ x == b])
