from abc import abstractmethod
from math import prod, sqrt
from typing import Iterable, Optional

import cvxpy as cp
import numpy as np
from scipy.fftpack import idct

from ..landscape.landscape import Landscape
from .base_reconstructor import BaseReconstructor


class BaseCvxPyReconstructor(BaseReconstructor):
    def __init__(self, solver: Optional[str] = None, **solver_kwargs):
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def run(self, landscape: Landscape) -> np.ndarray:
        shape = landscape.shape
        x = cp.Variable(landscape.size)
        self._build_optimization_problem(
            self._build_idct_operator(shape)[landscape._sampled_indices],
            x,
            landscape.sampled_landscape,
        ).solve(self.solver, self.solver_kwargs)
        x = x.value.reshape(shape)
        for i in range(len(shape)):
            x = idct(x, norm="ortho", axis=i)
        return x

    def _build_idct_operator(self, shape: Iterable[int]) -> np.ndarray:
        idct_operators = [idct(np.identity(n), norm="ortho", axis=0) for n in shape]
        a = idct_operators[0]
        for b in idct_operators[1:]:
            a = np.kron(a, b)
        return a

    @abstractmethod
    def _build_optimization_problem(
        self, A: np.ndarray, x: cp.Variable, b: np.ndarray
    ) -> cp.Problem:
        pass

    def _reshape_2D(self, shape: Iterable[int]) -> tuple[int, int]:
        """
        Reshape higher dimensions to 2D.
        No longer needed (hopefully).
        """
        # partition the arbitrary dimension shape into approximately two equal parts
        # naive solution - can be improved with transpose and dynamic programming but not really necessary
        size = prod(shape)
        sqrt_size = sqrt(size)
        num_row = 1
        for i, s in enumerate(shape):
            if num_row * s > sqrt_size:
                break
            num_row *= s
        num_col = prod(shape[i:])
        return num_row, num_col
