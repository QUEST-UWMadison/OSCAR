from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from functools import reduce
from math import prod, sqrt
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.fftpack import idct

if TYPE_CHECKING:
    from ..landscape.landscape import Landscape

from .base_reconstructor import BaseReconstructor


class BaseCvxPyReconstructor(BaseReconstructor):
    def __init__(self, solver: str | None = None, **solver_kwargs) -> None:
        self.solver: str | None = solver
        self.solver_kwargs: dict[str, Any] = solver_kwargs

    def run(self, landscape: Landscape) -> NDArray[np.float64]:
        if landscape.sampled_landscape is None:
            raise RuntimeError(
                "Sampled landscape is not present. Use `Landscape.sample_and_run()`, "
                "`Landscape.run_with_indices()`, or `Landscape.run_with_flatten_indices()`."
            )
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

    def _build_idct_operator(self, shape: Sequence[int]) -> NDArray[np.float64]:
        return reduce(np.kron, (idct(np.identity(n), norm="ortho", axis=0) for n in shape))

    @abstractmethod
    def _build_optimization_problem(
        self, A: NDArray[np.float64], x: cp.Variable, b: NDArray[np.float64]
    ) -> cp.Problem:
        pass

    def _reshape_2D(self, shape: Sequence[int]) -> tuple[int, int]:
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
