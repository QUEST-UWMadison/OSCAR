from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .base_executor import BaseExecutor


class CustomExecutor(BaseExecutor):
    def __init__(
        self,
        function: Callable[[Sequence[float]], float],
        batch_function: Callable[[Sequence[Sequence[float]]], Iterable[float]] | None = None,
    ) -> None:
        self.function: Callable[[Sequence[float]], float] = function
        self.batch_function: Callable[
            [Sequence[Sequence[float]]], Iterable[float]
        ] | None = batch_function

    def _run(self, params: Sequence[float], **kwargs) -> float:
        return self.function(params, **kwargs)

    def run_batch(
        self,
        params_list: Iterable[Sequence[float]],
        callback: Callable[[Sequence[float], float, float], None] | None = None,
        **kwargs,
    ) -> Iterable[float]:
        if self.batch_function is None:
            return super().run_batch(params_list, callback)
        return self.batch_function(params_list, **kwargs)
