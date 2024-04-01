from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import teneva
from numpy.typing import NDArray

from ..landscape.landscape_data import TensorNetworkLandscapeData

if TYPE_CHECKING:
    from ..landscape.landscape import Landscape

from .base_reconstructor import BaseReconstructor


class TenevaReconstructor(BaseReconstructor):
    def __init__(
        self,
        anova_rank: int = 2,
        anova_order: Literal[1, 2] = 1,
        anova_noise: float = 1.0e-10,
        als_regularization: float = 0.001,
        als_nsweeps: int = 50,
        als_convergence: float = 1.0e-10,
        als_max_rank: int | None = None,
        als_weights: Sequence[float] | None = None,
        validation_fraction: float = 0.1,
        validation_convergence: float = 1.0e-10,
        seed: int | np.random.Generator = None,
    ) -> None:
        self.anova_rank: int = anova_rank
        self.anova_order: Literal[1, 2] = anova_order
        self.anova_noise: float = anova_noise
        self.als_regularization: float = als_regularization
        self.als_nsweeps: int = als_nsweeps
        self.als_convergence: float = als_convergence
        self.als_max_rank: int | None = als_max_rank
        self.als_weights: Sequence[float] | None = als_weights
        self.validation_fraction: float = validation_fraction
        self.validation_convergence: float = validation_convergence
        self.seed: int | np.random.Generator = seed
        self.result: dict[str, Any] = {}
        super().__init__()

    def run(
        self,
        landscape: Landscape,
        verbose: bool = False,
        callback: Callable[[list[NDArray[np.float64]], dict, dict], bool | None] = None,
    ) -> NDArray[np.float64]:
        if landscape.sampled_landscape is None:
            raise RuntimeError(
                "Sampled landscape is not present. Use `Landscape.sample_and_run()`, "
                "`Landscape.run_with_indices()`, or `Landscape.run_with_flatten_indices()`."
            )

        sampled_indices = np.asarray(landscape.sampled_indices_unravelled).T
        tensors = teneva.anova(
            sampled_indices,
            landscape.sampled_landscape,
            r=self.anova_rank,
            order=self.anova_order,
            noise=self.anova_noise,
            seed=self.seed,
        )
        rng = np.random.default_rng(self.seed)
        shuffle = rng.permutation(landscape.num_samples)
        data_train = landscape.sampled_landscape[shuffle]
        indices_train = sampled_indices[shuffle]
        split = int(landscape.num_samples * self.validation_fraction)
        if split > 0:
            data_valid = data_train[:split]
            indices_valid = indices_train[:split]
            data_train = data_train[split:]
            indices_train = indices_train[split:]
        else:
            data_valid = None
            indices_valid = None
        if self.als_nsweeps > 0:
            tensors = teneva.als(
                indices_train,
                data_train,
                tensors,
                nswp=self.als_nsweeps,
                e=self.als_convergence,
                info=self.result,
                I_vld=indices_valid,
                y_vld=data_valid,
                e_vld=self.validation_convergence,
                r=self.als_max_rank,
                lamb=self.als_regularization,
                w=self.als_weights,
                cb=callback,
                log=verbose,
            )
        return TensorNetworkLandscapeData(tensors)
