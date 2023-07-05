from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pdfo import OptimizeResult
    from SQCommon import Result
from qiskit.algorithms.optimizers import OptimizerResult


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
