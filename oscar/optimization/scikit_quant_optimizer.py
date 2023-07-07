from __future__ import annotations

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
        def __init__(
            self, method: Literal["imfil", "bobyqa", "snobfit", "nomad"], budget: int = 10000
        ) -> None:
            self.method: Literal["imfil", "bobyqa", "snobfit", "nomad"] = method
            self.budget: int = budget

        def name(self, include_library_name: bool = True) -> str:
            name: str = self.method
            if include_library_name:
                name += " (Scikit-Quant)"
            return name

        def run(
            self,
            executor: BaseExecutor,
            initial_point: Sequence[float],
            bounds: Sequence[Sequence[float]],  # TODO nomad
            *args,
            **kwargs,
        ) -> tuple[Trace, Result]:
            trace = Trace()
            result = minimize(
                partial(executor.run, callback=trace.append),
                np.array(initial_point),
                np.array(bounds),
                self.budget,
                self.method,
                *args,
                **kwargs,
            )
            trace.update_with_skquant_result(result)
            return trace, result

except:
    pass
