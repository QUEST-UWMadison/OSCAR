from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .base_optimizer import BaseOptimizer, ConstraintsType, JacobianType

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor, CallbackType


class CustomOptimizer(BaseOptimizer):
    def __init__(
        self, optimizer: Callable[[NDArray[np.float_]], Any], name: str | None = None
    ) -> None:
        self.optimizer: Callable[[NDArray[np.float_]], Any] = optimizer
        if name is None:
            name = self.optimizer.__name__
        self._name: str = name
        super().__init__()

    def name(self, include_library_name: bool = True) -> str:
        return self._name + (" (Custom)" if include_library_name else "")

    def _run(
        self,
        executor: BaseExecutor,
        initial_point: NDArray[np.float_],
        bounds: NDArray[np.float_] | None = None,
        jacobian: JacobianType | None = None,
        constraints: ConstraintsType | None = None,
        callback: CallbackType | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.result = self.optimizer(
            self._objective(executor, callback, **executor_kwargs),
            initial_point,
            bounds,
            jacobian,
            constraints,
            **kwargs,
        )
