from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .base_executor import BaseExecutor


class CustomExecutor(BaseExecutor):
    def __init__(
        self,
        function: Callable[[Sequence[float]], float],
        batch_function: Callable[[Sequence[Sequence[float]]], Sequence[float]] | None = None,
    ) -> None:
        self.function: Callable[[Sequence[float]], float] = function
        self.batch_function: Callable[
            [Sequence[Sequence[float]]], Sequence[float]
        ] | None = batch_function

    def _run(self, params: Sequence[float], *args, **kwargs) -> float:
        return self.function(params, *args, **kwargs)

    def run_batch(
        self,
        params_list: Sequence[Sequence[float]],
        return_time: bool = False,
        callback: Callable[[Sequence[float], float, float], None] | None = None,
        *args,
        **kwargs,
    ) -> NDArray[np.float_] | tuple[NDArray[np.float_], NDArray[np.float_] | None]:
        if self.batch_function is None:
            return super().run_batch(params_list, callback)
        result = np.array(self.batch_function(params_list, *args, **kwargs))
        if return_time:
            return result, None
        return result
