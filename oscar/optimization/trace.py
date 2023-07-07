from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from nlopt import opt
from numpy.typing import NDArray
from qiskit.algorithms.optimizers import OptimizerResult

if TYPE_CHECKING:
    from pdfo import OptimizeResult
    from SQCommon import Result


@dataclass
class Trace:
    params_trace: list[NDArray[np.float_]] = field(default_factory=list)
    value_trace: list[float] = field(default_factory=list)
    time_trace: list[float] = field(default_factory=list)
    optimal_params: NDArray[np.float_] | None = None
    optimal_value: float | None = None
    num_iters: int | None = None
    num_fun_evals: int | None = None

    def append(self, params: Sequence[float], value: float, time: float) -> None:
        self.params_trace.append(np.array(params))
        self.value_trace.append(value)
        self.time_trace.append(time)

    def print_result(self) -> None:
        print(f"Total time: {sum(self.time_trace)}")
        print("Optimal parameters reported: ", self.optimal_params)
        print("Optimal value reported: ", self.optimal_value)
        print("Number of evaluations: ", self.num_fun_evals)

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
