from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class Trace:
    def __init__(self, optimization_type: Literal["min", "max"] = "min") -> None:
        self.params_trace: list[NDArray[np.float64]] = []
        self.value_trace: list[float] = []
        self.time_trace: list[float] = []
        self.optimization_type: Literal["min", "max"] = optimization_type

    @property
    def num_fun_evals(self) -> int:
        return len(self.value_trace)

    @property
    def optimal_params(self) -> NDArray[np.float64]:  # Known limitation: may violate constraints
        argopt = np.argmin if self.optimization_type == "min" else np.argmax
        return self.params_trace[argopt(self.value_trace)]

    @property
    def optimal_value(self) -> float:  # Known limitation: may violate constraints
        return min(self.value_trace) if self.optimization_type == "min" else max(self.value_trace)

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
        return trace

    def __getitem__(self, axes: int | Sequence[int]) -> Trace:
        return self.project_to(axes)

    def __repr__(self) -> str:
        return (
            f"Total evaluation time: {self.total_time}"  # Known limitation: doesn't include optimizer time
            + f"\nOptimal parameters reported: {self.optimal_params}"
            + f"\nOptimal value reported: {self.optimal_value}"
            + f"\nNumber of evaluations: {self.num_fun_evals}"  # Known limitation: doesn't include Jacobian evaluations
        )
