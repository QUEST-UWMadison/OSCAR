from typing import Callable, Optional

import numpy as np

from .base_executor import BaseExecutor


class CustomExecutor(BaseExecutor):
    def __init__(
        self,
        function: Callable[[np.ndarray], float],
        batch_function: Optional[Callable[[list[np.ndarray]], float]] = None,
    ):
        self.function: Callable[[np.ndarray], float] = function
        self.batch_function: Callable[[list[np.ndarray]], float] | None = batch_function

    def _run(self, params: np.ndarray, *args, **kwargs) -> float:
        return self.function(params, *args, **kwargs)

    def run_batch(
        self,
        params_list: list[np.ndarray],
        callback: Callable[[np.ndarray, float, float], None] | None = None,
        *args,
        **kwargs
    ) -> np.ndarray:
        if self.batch_function is None:
            return super().run_batch(params_list, callback)
        return self.batch_function(params_list, *args, **kwargs)
