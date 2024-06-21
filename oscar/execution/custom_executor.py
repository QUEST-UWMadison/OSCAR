from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from .base_executor import BaseExecutor, CallbackType


class CustomExecutor(BaseExecutor):
    def __init__(
        self,
        function: Callable[[Sequence[float]], float],
        batch_function: Callable[[Iterable[Sequence[float]]], Iterable[float]] | None = None,
    ) -> None:
        self.function: Callable[[Sequence[float]], float] = function
        self.batch_function: Callable[[Iterable[Sequence[float]]], Iterable[float]] | None = (
            batch_function
        )

    def _run(self, params: Sequence[float], **kwargs) -> float:
        return self.function(params, **kwargs)

    def run_batch(
        self,
        params_list: Iterable[Sequence[float]],
        callback: CallbackType | None = None,
        **kwargs,
    ) -> Iterable[float]:
        if self.batch_function is None:
            return super().run_batch(params_list, callback)
        return self.batch_function(params_list, **kwargs)
