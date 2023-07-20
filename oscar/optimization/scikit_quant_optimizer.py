from __future__ import annotations

from typing import Any

try:
    from collections.abc import Sequence
    from functools import partial
    from typing import TYPE_CHECKING, Literal

    import numpy as np
    from skquant.opt import minimize
    from SQCommon import Result

    from .base_optimizer import BaseOptimizer
    from .trace import Trace

    if TYPE_CHECKING:
        from ..execution.base_executor import BaseExecutor

    class ScikitQuantOptimizer(BaseOptimizer):
        def __init__(self, method: Literal["imfil", "bobyqa", "snobfit", "nomad"]) -> None:
            self.method: Literal["imfil", "bobyqa", "snobfit", "nomad"] = method

        def name(self, include_library_name: bool = True) -> str:
            name: str = self.method
            if include_library_name:
                name += " (Scikit-Quant)"
            return name

        def run(
            self,
            executor: BaseExecutor,
            initial_point: Sequence[float],
            executor_kwargs: dict[str, Any] | None = None,
            bounds: Sequence[Sequence[float]] | None = None,  # TODO nomad
            budget: int = 10000,
            **kwargs,
        ) -> tuple[Trace, Result]:
            if executor_kwargs is None:
                executor_kwargs = {}
            trace = Trace()
            result = minimize(
                func=partial(executor.run, callback=trace.append, **executor_kwargs),
                x0=np.array(initial_point),
                bounds=None if bounds is None else np.array(bounds),
                budget=budget,
                method=self.method,
                **kwargs,
            )
            trace.update_with_skquant_result(result)
            return trace, result

except ImportError:
    pass
