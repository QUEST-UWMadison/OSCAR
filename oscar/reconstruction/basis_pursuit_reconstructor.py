from __future__ import annotations

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .base_cvxpy_reconstructor import BaseCvxPyReconstructor


class BPReconstructor(BaseCvxPyReconstructor):
    def _build_optimization_problem(
        self, A: NDArray[np.float_], x: cp.Variable, b: NDArray[np.float_]
    ) -> cp.Problem:
        return cp.Problem(cp.Minimize(cp.norm(x, 1)), [A @ x == b])
