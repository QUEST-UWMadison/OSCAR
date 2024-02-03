from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quimb
import quimb.tensor as qtn
import teneva
from numpy.typing import NDArray
from scipy.fft import idct

from ..landscape.landscape_data import TensorNetworkLandscapeData

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

        sampled_indices = np.asarray(landscape.unravel_index(landscape.sampled_indices)).T
        tensors = teneva.anova(
            sampled_indices,
            landscape.sampled_landscape,
            **self.anova_args,
        )
        tensors = teneva.als(
            sampled_indices, landscape.sampled_landscape, tensors, **self.als_args
        )
        # return teneva.get_many(
        #     tensors, np.asarray(landscape._unravel_index(np.arange(landscape.size))).T
        # ).reshape(landscape.shape)
        return TensorNetworkLandscapeData(tensors)

        # tensors[0] = tensors[0].reshape(tensors[0].shape[1:])
        # tensors[-1] = tensors[-1].reshape(tensors[-1].shape[:-1])
        # dmrg = qtn.DMRGX(A, bond_dims=64, cutoffs=1e-10, p0=b)
        # dmrg.solve(verbosity=1)
        # x = dmrg.state.to_dense().reshape(shape)
        for i in range(len(shape)):
            x = idct(x, norm="ortho", axis=i)
        return x
