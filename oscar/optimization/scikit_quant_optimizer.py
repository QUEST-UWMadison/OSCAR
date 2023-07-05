from __future__ import annotations

try:
    from collections.abc import Sequence
    from functools import partial
    from typing import Literal

    import numpy as np
    from skquant.opt import minimize
    from SQCommon import Result

    from ..execution.base_executor import BaseExecutor
    from .base_optimizer import BaseOptimizer
    from .trace import Trace

    class ScikitQuantOptimizer(BaseOptimizer):
        def __init__(
            self, method: Literal["imfil", "bobyqa", "snobfit", "nomad"], budget: int = 10000
        ) -> None:
            self.method: Literal["imfil", "bobyqa", "snobfit", "nomad"] = method
            self.budget: int = budget

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
