import pprint

import numpy as np
from qiskit.algorithms.optimizers import OptimizerResult


class Trace:
    def __init__(self):
        self.params_trace: list[np.ndarray] = []
        self.value_trace: list[float] = []
        self.time_trace: list[float] = []
        self.optimal_params: np.ndarray | None = None
        self.optimal_value: float | None = None
        self.num_iters: int = 0
        self.num_fun_evals: int = 0

    def __str__(self) -> str:
        return pprint.pformat(self.__dict__, indent=4)

    def append(self, params: np.ndarray, value: float, time: float) -> None:
        self.params_trace.append(params)
        self.value_trace.append(value)
        self.time_trace.append(time)

    def update_with_qiskit_result(self, result: OptimizerResult) -> None:
        self.optimal_params = result.x
        self.optimal_value = result.fun
        self.num_iters = result.nit
        self.num_fun_evals = result.nfev
