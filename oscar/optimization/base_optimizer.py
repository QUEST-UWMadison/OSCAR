from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .trace import Trace

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor, CallbackType

ObjectiveType: TypeAlias = Callable[[NDArray[np.float64]], float]
JacobianType: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.float64]]
ConstraintsType: TypeAlias = Sequence[
    tuple[Literal["eq", "ineq"], ObjectiveType, JacobianType | None]
    | tuple[Literal["eq", "ineq"], ObjectiveType]
]


class BaseOptimizer(ABC):
    def __init__(self) -> None:
        self.result: Any = None
        self.trace: Trace | None = None
        super().__init__()

    @abstractmethod
    def name(self, include_library_name: bool = True) -> str:
        pass

    def run(
        self,
        executor: BaseExecutor,
        initial_point: Sequence[float],
        bounds: Sequence[Sequence[float]] | None = None,
        jacobian: JacobianType | None = None,
        constraints: ConstraintsType | None = None,
        callback: CallbackType | None = None,
        **executor_kwargs,
    ) -> tuple[Trace, Any]:
        self.trace = Trace()
        try:
            self._run(
                executor,
                np.asarray(initial_point),
                None if bounds is None else np.asarray(bounds),
                jacobian,
                constraints,
                callback,
                **executor_kwargs,
            )
        except KeyboardInterrupt:
            pass
        return self.trace, self.result

    def _objective(
        self,
        executor: BaseExecutor,
        callback: CallbackType | None = None,
        **executor_kwargs,
    ) -> ObjectiveType:
        def callback_wrapper(params: NDArray[np.float64], value: float, runtime: float) -> None:
            self.trace.append(params, value, runtime)
            if callback is not None:
                if callback(params, value, runtime):
                    raise KeyboardInterrupt

        return partial(executor.run, callback=callback_wrapper, **executor_kwargs)

    @abstractmethod
    def _run(
        self,
        executor: BaseExecutor,
        initial_point: NDArray[np.float64],
        bounds: NDArray[np.float64] | None = None,
        jacobian: JacobianType | None = None,  # TODO: make jacobian part of executor
        constraints: ConstraintsType | None = None,
        callback: CallbackType | None = None,
        **executor_kwargs,
    ) -> None:
        pass
