from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import partial, reduce, singledispatchmethod
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct, idct

from ..execution import CustomExecutor
from ..landscape.landscape_data import TensorLandscapeData
from ..optimization import Trace
from ..optimization.base_optimizer import BaseOptimizer
from .base_reconstructor import BaseReconstructor

if TYPE_CHECKING:
    from ..landscape.landscape import Landscape


class CSReconstructor(BaseReconstructor):
    def __init__(
        self,
        regularization: float | None = 0.001,
        tolerance: float = 1.0e-6,
        norm: int | Literal["fro", "inf", "nuc", "tv"] = 1,
        reverse_objectives: bool = False,
        solver: str | BaseOptimizer | None = None,
    ) -> None:
        self.regularization: float = regularization
        self.tolerance: float = tolerance
        self.norm: int | Literal["fro", "inf", "nuc", "tv"] = norm
        self.reverse_objectives: bool = reverse_objectives
        self.solver: str | BaseOptimizer | None = solver
        self.result: cp.Problem | Trace | None = None

    @property
    def backend(self) -> ModuleType:
        return np if isinstance(self.solver, BaseOptimizer) else cp

    def run(
        self,
        landscape: Landscape,
        verbose: bool = False,
        callback: Callable[[Any], None] | None = None,
        **solver_kwargs,
    ) -> NDArray[np.float64]:
        sampled_landscape = landscape.sampled_landscape
        if sampled_landscape is None:
            raise RuntimeError(
                "Sampled landscape is not present. Use `Landscape.sample_and_run()`, "
                "`Landscape.run_with_indices()`, or `Landscape.run_with_flatten_indices()`."
            )
        sampled_indices = landscape.sampled_indices
        shape = landscape.shape
        objective, constraint = self.objective, self.constraint
        objective_jacobian, constraint_jacobian = self.objective_jacobian, self.constraint_jacobian
        if self.reverse_objectives:
            objective, constraint = constraint, objective
            objective_jacobian, constraint_jacobian = constraint_jacobian, objective_jacobian

        if isinstance(self.solver, BaseOptimizer):
            if landscape.landscape is None:
                x = np.random.default_rng().normal(scale=1.0e-6, size=landscape.size)
                x[sampled_indices] = sampled_landscape
            else:
                x = landscape.landscape.to_numpy()
            x = self._project_to_basis(x.reshape(shape), False).reshape(-1)
            executor = CustomExecutor(
                partial(
                    objective,
                    b=sampled_landscape,
                    sampled_indices=sampled_indices,
                    shape=shape,
                )
            )
            self.result = self.solver.run(
                executor,
                initial_point=x,
                jacobian=partial(
                    objective_jacobian,
                    b=sampled_landscape,
                    sampled_indices=sampled_indices,
                    shape=shape,
                ),
                constraints=(
                    [
                        (
                            "ineq",
                            partial(
                                constraint,
                                b=sampled_landscape,
                                sampled_indices=sampled_indices,
                                shape=shape,
                            ),
                            partial(
                                constraint_jacobian,
                                b=sampled_landscape,
                                sampled_indices=sampled_indices,
                                shape=shape,
                            ),
                        )
                    ]
                    if self.regularization is None
                    else None
                ),
                callback=callback,
                **solver_kwargs,
            )[0]
            x = self.result.optimal_params
        else:
            if callback is not None:
                warnings.warn("The `callback` argument is ignored for the cvxpy backend.")
            x = cp.Variable(landscape.size)
            self.result = cp.Problem(
                cp.Minimize(objective(x, sampled_landscape, sampled_indices, shape)),
                (
                    None
                    if self.regularization
                    else [constraint(x, sampled_landscape, sampled_indices, shape) >= 0]
                ),
            )
            self.result.solve(self.solver, verbose=verbose, **solver_kwargs)
            x = x.value
        return TensorLandscapeData(self._project_to_basis(x.reshape(shape)))

    def objective(
        self,
        x: NDArray[np.float64] | cp.Expression,
        b: NDArray[np.float64],
        sampled_indices: NDArray[np.int_],
        shape: Sequence[int],
    ) -> cp.Expression | float:
        objective = self._norm(x.reshape(shape))
        if self.regularization is None:
            return objective
        return (
            self.regularization * objective
            - self.constraint(x, b, sampled_indices, shape)
            + self.tolerance
        )

    def constraint(
        self,
        x: NDArray[np.float64] | cp.Expression,
        b: NDArray[np.float64],
        sampled_indices: NDArray[np.int_],
        shape: Sequence[int],
    ) -> float | cp.Expression:
        return self.tolerance - (
            self._l2_norm(
                self._project_to_basis(x.reshape(shape)).reshape(-1)[sampled_indices] - b
            )
            / self._l2_norm(b)
        )

    def objective_jacobian(
        self,
        x: NDArray[np.float64],
        b: NDArray[np.float64],
        sampled_indices: NDArray[np.int_],
        shape: Sequence[int],
    ) -> NDArray[np.float64]:
        if self.norm != 1:
            raise NotImplementedError("Jacobian is not implemented for norms other than 1.")
        objective = np.sign(x)
        if self.regularization is None:
            return objective
        return self.regularization * objective - self.constraint_jacobian(
            x, b, sampled_indices, shape
        )

    def constraint_jacobian(
        self,
        x: NDArray[np.float64],
        b: NDArray[np.float64],
        sampled_indices: NDArray[np.int_],
        shape: Sequence[int],
    ) -> NDArray[np.float64]:
        diff = np.zeros_like(x)
        diff[sampled_indices] = (
            self._project_to_basis(x.reshape(shape)).reshape(-1)[sampled_indices] - b
        )
        return (
            -2
            * self._project_to_basis(diff.reshape(shape), inverse=False).reshape(-1)
            / self._l2_norm(b)
        )

    @singledispatchmethod
    def _norm(self, x: NDArray[np.float64]) -> float:
        if self.norm == "tv":
            return np.sum(np.abs([np.diff(x, axis=i) for i in range(x.ndim)]))
        return np.linalg.norm(x.reshape(-1), ord=self.norm)

    @_norm.register
    def _norm_cp(self, x: cp.Expression) -> cp.Expression:
        if self.norm == "tv":
            return cp.tv(x)
        return cp.norm(x.reshape(-1), p=self.norm)

    @singledispatchmethod
    def _l2_norm(self, x: NDArray[np.float64]) -> float:
        return np.linalg.norm(x, 2)

    @_l2_norm.register
    def _l2_norm_cp(self, x: cp.Expression) -> cp.Expression:
        return cp.norm(x, 2)

    @singledispatchmethod
    def _project_to_basis(
        self, x: NDArray[np.float64], inverse: bool = True
    ) -> NDArray[np.float64]:
        project = idct if inverse else dct
        for i in range(len(x.shape)):
            x = project(x, norm="ortho", axis=i)
        return x

    @_project_to_basis.register
    def _project_to_basis_cp(self, x: cp.Expression, inverse: bool = True) -> cp.Expression:
        project = idct if inverse else dct
        return reduce(
            np.kron, (project(np.identity(n), norm="ortho", axis=0) for n in x.shape)
        ) @ x.reshape(-1)
