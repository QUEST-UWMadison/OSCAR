import pprint
from dataclasses import dataclass, field

import numpy as np
from qiskit.algorithms.optimizers import OptimizerResult


@dataclass
class Trace:
    params_trace: list[np.ndarray] = field(default_factory=list)
    value_trace: list[float] = field(default_factory=list)
    time_trace: list[float] = field(default_factory=list)
    optimal_params: np.ndarray | None = None
    optimal_value: float | None = None
    num_iters: int = 0
    num_fun_evals: int = 0

    # def __str__(self) -> str:
    #     return pprint.pformat(self.__dict__, indent=4)

    def append(self, params: np.ndarray, value: float, time: float) -> None:
        self.params_trace.append(params)
        self.value_trace.append(value)
        self.time_trace.append(time)

    def update_with_qiskit_result(self, result: OptimizerResult) -> None:
        self.optimal_params = result.x
        self.optimal_value = result.fun
        self.num_iters = result.nit
        self.num_fun_evals = result.nfev
