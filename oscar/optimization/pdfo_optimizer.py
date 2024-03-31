from __future__ import annotations

import warnings
from typing import Any

try:
    from typing import TYPE_CHECKING, Literal

    import numpy as np
    import pdfo
    from numpy.typing import NDArray

    from .base_optimizer import BaseOptimizer, ConstraintsType, JacobianType
    from .scipy_optimizer import to_scipy_constraints

    if TYPE_CHECKING:
        from ..execution.base_executor import BaseExecutor, CallbackType

    class PDFOOptimizer(BaseOptimizer):
        def __init__(
            self,
            optimizer: Literal["uobyqa", "newuoa", "bobyqa", "lincoa", "cobyla"] | None,
            **optimizer_kwargs,
        ) -> None:
            self.optimizer: Literal["uobyqa", "newuoa", "bobyqa", "lincoa", "cobyla"] | None = optimizer
            self.optimizer_kwargs: dict[str, Any] = optimizer_kwargs

        def name(self, include_library_name: bool = True) -> str:
            name = "default" if self.optimizer is None else str(self.optimizer)
            if include_library_name:
                name += " (PDFO)"
            return name

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
            if jacobian is not None:
                warnings.warn("Jacobian is ignored for Scikit-Quant methods.")
            self.result = pdfo.pdfo(
                fun=self._objective(executor, callback, **executor_kwargs),
                x0=np.array(initial_point),
                method=self.optimizer,
                bounds=None if bounds is None else np.array(bounds),
                constraints=to_scipy_constraints(constraints),
                options=self.optimizer_kwargs,
            )

except ImportError:
    pass
