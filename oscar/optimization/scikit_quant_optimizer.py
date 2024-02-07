from __future__ import annotations

try:
    import warnings
    from typing import TYPE_CHECKING, Any, Literal

    import numpy as np
    from numpy.typing import NDArray
    from skquant.opt import minimize  # type: ignore

    from .base_optimizer import BaseOptimizer, ConstraintsType, JacobianType

    if TYPE_CHECKING:
        from ..execution.base_executor import BaseExecutor, CallbackType

    class ScikitQuantOptimizer(BaseOptimizer):
        def __init__(
            self, method: Literal["imfil", "bobyqa", "snobfit", "nomad"], budget: int
        ) -> None:
            self.method: Literal["imfil", "bobyqa", "snobfit", "nomad"] = method
            self.budget: int = budget
            super().__init__()

        def name(self, include_library_name: bool = True) -> str:
            name: str = self.method
            if include_library_name:
                name += " (Scikit-Quant)"
            return name

        def _run(
            self,
            executor: BaseExecutor,
            initial_point: NDArray[np.float64],
            bounds: NDArray[np.float64] | None = None,
            jacobian: JacobianType | None = None,
            constraints: ConstraintsType | None = None,
            callback: CallbackType | None = None,
            executor_kwargs: dict[str, Any] | None = None,
            **kwargs,
        ) -> None:
            if jacobian is not None:
                warnings.warn("Jacobian is ignored for Scikit-Quant methods.")
            if constraints is not None:
                warnings.warn("Constraints are ignored for Scikit-Quant methods.")
            self.result = minimize(
                func=self._objective(executor, callback, **executor_kwargs),
                x0=initial_point,
                bounds=None if bounds is None else np.array(bounds),
                budget=self.budget,
                method=self.method,
                **kwargs,
            )

except ImportError:
    pass
