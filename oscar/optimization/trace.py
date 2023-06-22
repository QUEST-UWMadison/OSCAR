from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from qiskit.algorithms.optimizers import OptimizerResult
from SQCommon import Result, Stats


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
