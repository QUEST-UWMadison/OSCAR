from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor
    from .trace import Trace


def _evaluate_optimal_point(trace: Trace, executor: BaseExecutor | None = None) -> float:
    if executor is None:
        return trace.optimal_value
    return executor.run(trace.optimal_params)


def approximation_ratio(
    min_value: float, max_value: float, executor: BaseExecutor | None = None
) -> Callable[[str, int, Trace, Any], float]:
    def compute_ar(config_set_index: str, config_index: int, trace: Trace, result: Any) -> float:
        return (_evaluate_optimal_point(trace, executor) - min_value) / (max_value - min_value)

    return compute_ar


def optimal_value(executor: BaseExecutor | None = None) -> Callable[[str, int, Trace, Any], float]:
    def get_optimal_value(config_set_index: str, config_index: int, trace: Trace, result: Any) -> float:
        return _evaluate_optimal_point(trace, executor)

    return get_optimal_value
