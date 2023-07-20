from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from nlopt import opt
from numpy.typing import NDArray

from .base_optimizer import BaseOptimizer
from .trace import Trace

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor


class NLoptOptimizer(BaseOptimizer):
    @singledispatchmethod
    def __init__(self, optimizer: opt | str) -> None:
        self.optimizer: opt | str = optimizer

    def name(self, include_library_name: bool = True) -> str:
        if isinstance(self.optimizer, str):
            name = self.optimizer
        else:
            name = self.optimizer.get_algorithm_name()
        if include_library_name:
            name += " (NLopt)"
        return name

    def run(
        self,
        executor: BaseExecutor,
        initial_point: Sequence[float],
        executor_kwargs: dict[str, Any] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        equality_constraints: Sequence[
            tuple[Callable[[Sequence[float], NDArray[np.float_]], float], float]
        ]
        | Sequence[
            tuple[
                Callable[[NDArray[np.float_], Sequence[float], NDArray[np.float_]], float],
                NDArray[np.float_],
            ]
        ]
        | None = None,
        inequality_constraints: Sequence[
            tuple[Callable[[Sequence[float], NDArray[np.float_]], float], float]
        ]
        | Sequence[
            tuple[
                Callable[[NDArray[np.float_], Sequence[float], NDArray[np.float_]], float],
                NDArray[np.float_],
            ]
        ]
        | None = None,
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
        **kwargs,
    ) -> tuple[Trace, opt]:
        if executor_kwargs is None:
            executor_kwargs = {}
        trace = Trace()

        def objective_wrapper(params: Sequence[float], gradient: NDArray[np.float_]) -> float:
            if gradient.size > 0:
                raise NotImplementedError("NLopt gradient-based algorithms are not supported yet.")
            return executor.run(params, callback=trace.append, **executor_kwargs)

        if isinstance(self.optimizer, str):
            self.optimizer = opt(self.optimizer, len(initial_point))
        if bounds is not None:
            bounds = np.array(bounds).T
            self.optimizer.set_lower_bounds(bounds[0])
            self.optimizer.set_upper_bounds(bounds[1])
        if equality_constraints is not None:
            for constraint in equality_constraints:
                if isinstance(constraint[1], np.ndarray):
                    self.optimizer.add_equality_mconstraint(*constraint)
                else:
                    self.optimizer.add_equality_constraint(*constraint)
        if inequality_constraints is not None:
            for constraint in equality_constraints:
                if isinstance(constraint[1], np.ndarray):
                    self.optimizer.add_inequality_mconstraint(*constraint)
                else:
                    self.optimizer.add_inequality_constraint(*constraint)
        if stopval is not None:
            self.optimizer.set_stopval(stopval)
        if ftol_rel is not None:
            self.optimizer.set_ftol_rel(ftol_rel)
        if ftol_abs is not None:
            self.optimizer.set_ftol_abs(ftol_abs)
        if xtol_rel is not None:
            self.optimizer.set_xtol_rel(xtol_rel)
        if xtol_abs is not None:
            self.optimizer.set_xtol_abs(xtol_abs)
        if x_weights is not None:
            self.optimizer.set_x_weights(x_weights)
        if maxeval is not None:
            self.optimizer.set_maxeval(maxeval)
        if maxtime is not None:
            self.optimizer.set_maxtime(maxtime)
        if local_optimizer is not None:
            self.optimizer.set_local_optimizer(local_optimizer)
        if initial_step is not None:
            self.optimizer.set_initial_step(initial_step)
        if population is not None:
            self.optimizer.set_population(population)
        if vector_storage is not None:
            self.optimizer.set_vector_storage(vector_storage)
        for key, valule in kwargs.items():
            self.optimizer.set_param(key, valule)

        self.optimizer.set_min_objective(objective_wrapper)
        optimal_params = self.optimizer.optimize(np.array(initial_point))
        trace.update_with_nlopt_result(self.optimizer, optimal_params)
        return trace, self.optimizer
