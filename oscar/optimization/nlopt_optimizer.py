from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from nlopt import opt
from numpy.typing import NDArray

from .base_optimizer import BaseOptimizer, ConstraintsType, JacobianType, ObjectiveType

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor, CallbackType


class NLoptOptimizer(BaseOptimizer):
    def __init__(
        self,
        optimizer: opt | str,
        stopval: float | None = None,
        ftol_rel: float | None = None,
        ftol_abs: float | None = None,
        xtol_rel: float | None = None,
        xtol_abs: float | None = None,
        x_weights: Sequence[float] | None = None,
        maxeval: int | None = None,
        maxtime: float | None = None,
        local_optimizer: opt | None = None,
        initial_step: Sequence[float] | float | None = None,
        population: int | None = None,
        vector_storage: int | None = None,
    ) -> None:
        self.optimizer: opt | str = optimizer
        self.stopval: float | None = stopval
        self.ftol_rel: float | None = ftol_rel
        self.ftol_abs: float | None = ftol_abs
        self.xtol_rel: float | None = xtol_rel
        self.xtol_abs: float | None = xtol_abs
        self.x_weights: Sequence[float] | None = x_weights
        self.maxeval: int | None = maxeval
        self.maxtime: float | None = maxtime
        self.local_optimizer: opt | None = local_optimizer
        self.initial_step: Sequence[float] | float | None = initial_step
        self.population: int | None = population
        self.vector_storage: int | None = vector_storage
        super().__init__()

    def name(self, include_library_name: bool = True) -> str:
        if isinstance(self.optimizer, str):
            name = self.optimizer
        else:
            name = self.optimizer.get_algorithm_name()
        if include_library_name:
            name += " (NLopt)"
        return name

    def _objective_factory(
        self,
        function: ObjectiveType,
        jacobian: JacobianType | None = None,
    ) -> ObjectiveType:
        def objective(params: NDArray[np.float_], gradient: NDArray[np.float_]) -> float:
            if jacobian is not None:
                if gradient.size > 0:
                    gradient[:] = jacobian(params)
                else:
                    warnings.warn(f"Jacobian is ignored for {self.name()}.")
            elif gradient.size > 0:
                raise ValueError(f"Jacobian is required for {self.name()}.")
            return function(params)

        return objective

    def _run(
        self,
        executor: BaseExecutor,
        initial_point: NDArray[np.float_],
        bounds: NDArray[np.float_] | None = None,
        jacobian: JacobianType | None = None,
        constraints: ConstraintsType | None = None,
        callback: CallbackType | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if isinstance(self.optimizer, str):
            self.optimizer = opt(self.optimizer, len(initial_point))

        if bounds is not None:
            self.optimizer.set_lower_bounds(bounds[:, 0])
            self.optimizer.set_upper_bounds(bounds[:, 1])

        if constraints is not None:
            for constraint in constraints:
                if len(constraint) == 2:
                    constraint = (*constraint, None)
                if constraint[0] == "eq":
                    add_constraint = self.optimizer.add_equality_constraint
                elif constraint[0] == "ineq":
                    add_constraint = self.optimizer.add_inequality_constraint
                else:
                    raise ValueError(f"Unknown constraint type: {constraint[0]}")
                add_constraint(self._objective_factory(*constraint[1:]), 1e-13)

        if self.stopval is not None:
            self.optimizer.set_stopval(self.stopval)
        if self.ftol_rel is not None:
            self.optimizer.set_ftol_rel(self.ftol_rel)
        if self.ftol_abs is not None:
            self.optimizer.set_ftol_abs(self.ftol_abs)
        if self.xtol_rel is not None:
            self.optimizer.set_xtol_rel(self.xtol_rel)
        if self.xtol_abs is not None:
            self.optimizer.set_xtol_abs(self.xtol_abs)
        if self.x_weights is not None:
            self.optimizer.set_x_weights(self.x_weights)
        if self.maxeval is not None:
            self.optimizer.set_maxeval(self.maxeval)
        if self.maxtime is not None:
            self.optimizer.set_maxtime(self.maxtime)
        if self.local_optimizer is not None:
            self.optimizer.set_local_optimizer(self.local_optimizer)
        if self.initial_step is not None:
            self.optimizer.set_initial_step(self.initial_step)
        if self.population is not None:
            self.optimizer.set_population(self.population)
        if self.vector_storage is not None:
            self.optimizer.set_vector_storage(self.vector_storage)
        for key, valule in kwargs.items():
            self.optimizer.set_param(key, valule)

        self.optimizer.set_min_objective(
            self._objective_factory(
                self._objective(executor, callback, **executor_kwargs), jacobian
            )
        )
        self.result = self.optimizer
        self.optimizer.optimize(initial_point)
