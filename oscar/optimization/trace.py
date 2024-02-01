from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from nlopt import opt
from numpy.typing import NDArray
from qiskit.algorithms.optimizers import OptimizerResult

if TYPE_CHECKING:
    from pdfo import OptimizeResult
    from SQCommon import Result


class Trace:
    def __init__(self) -> None:
        self.params_trace: list[NDArray[np.float_]] = []
        self.value_trace: list[float] = []
        self.time_trace: list[float] = []
        self.optimal_params: NDArray[np.float_] | None = None
        self.optimal_value: float | None = None
        self.num_iters: int | None = None
        self.num_fun_evals: int | None = None

    @property
    def total_time(self) -> float:
        return sum(self.time_trace)

    def append(self, params: Sequence[float], value: float, time: float) -> None:
        self.params_trace.append(np.array(params))
        self.value_trace.append(value)
        self.time_trace.append(time)

    def print_result(self) -> None:
        print(self)

    def project_to(self, *axes: int) -> Trace:
        if len(axes) == 1 and isinstance(axes[0], Sequence):
            axes = axes[0]
        axes = list(axes)
        trace = copy(self)
        trace.params_trace = [params[axes] for params in trace.params_trace]
        if trace.optimal_params is not None:
            trace.optimal_params = trace.optimal_params[axes]
        return trace

    def update_with_custom_result(self, result: Mapping[str, Any]) -> None:
        self.optimal_params = result["optimal_params"]
        self.optimal_value = result["optimal_value"]
        self.num_iters = result["num_iters"]
        self.num_fun_evals = result["num_fun_evals"]

    def update_with_nlopt_result(self, result: opt, optimal_params: NDArray[np.float_]) -> None:
        self.optimal_params = optimal_params
        self.optimal_value = result.last_optimum_value()
        self.num_iters = result.get_numevals()
        self.num_fun_evals = result.get_numevals()

    def update_with_qiskit_result(self, result: OptimizerResult) -> None:
        self.optimal_params = result.x
        self.optimal_value = result.fun
        self.num_iters = result.nit
        self.num_fun_evals = result.nfev

    def update_with_skquant_result(self, result: tuple[Result, NDArray[np.float_]]) -> None:
        self.optimal_params = result[0].optpar
        self.optimal_value = result[0].optval
        self.num_fun_evals = len(result[1])
        self.num_iters = len(result[1])

    def update_with_pdfo_result(self, result: OptimizeResult) -> None:
        self.optimal_params = result.x
        self.optimal_value = result.fun
        self.num_iters = result.nfev
        self.num_fun_evals = result.nfev

    def __getitem__(self, axes: int | Sequence[int]) -> Trace:
        return self.project_to(axes)

    def __repr__(self) -> str:
        return (
            f"Total time: {self.total_time}"
            + f"\nOptimal parameters reported: {self.optimal_params}"
            + f"\nOptimal value reported: {self.optimal_value}"
            + f"\nNumber of evaluations: {self.num_fun_evals}"
        )
