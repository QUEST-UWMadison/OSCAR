from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import teneva
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..landscape.landscape import Landscape

from .base_reconstructor import BaseReconstructor


class TenevaReconstructor(BaseReconstructor):
    def __init__(self, anova_args: dict | None = None, als_args: dict | None = None) -> None:
        self.anova_args: dict | None = {} if anova_args is None else anova_args
        self.als_args: dict | None = {} if als_args is None else als_args

    def run(self, landscape: Landscape) -> NDArray[np.float_]:
        if landscape.sampled_landscape is None:
            raise RuntimeError(
                "Sampled landscape is not present. Use `Landscape.sample_and_run()`, "
                "`Landscape.run_with_indices()`, or `Landscape.run_with_flatten_indices()`."
            )

        sampled_indices = np.array(landscape._unravel_index(landscape._sampled_indices)).T
        tensors = teneva.anova(
            sampled_indices,
            landscape.sampled_landscape,
            **self.anova_args,
        )
        tensors = teneva.als(
            sampled_indices, landscape.sampled_landscape, tensors, **self.als_args
        )
        # return teneva.get_many(
        #     tensors, np.array(landscape._unravel_index(np.arange(landscape.size))).T
        # ).reshape(landscape.shape)
        return teneva.full(tensors)
