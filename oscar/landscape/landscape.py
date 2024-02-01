from __future__ import annotations

import math
import os
import pickle
import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from ..execution.base_executor import BaseExecutor
from ..reconstruction import BPReconstructor
from .landscape_data import (
    LandscapeData,
    TensorLandscapeData,
    TensorNetworkLandscapeData,
)

if TYPE_CHECKING:
    from ..reconstruction.base_reconstructor import BaseReconstructor


class Landscape:
    def __init__(
        self,
        param_resolutions: Sequence[int],
        param_bounds: Sequence[tuple[float, float]]
        | tuple[float, float] = (
            -math.pi / 2,
            math.pi / 2,
        ),
    ) -> None:
        self.num_params: int = len(param_resolutions)
        self.param_resolutions: NDArray[np.int_] = np.array(param_resolutions)
        if not isinstance(param_bounds[0], Iterable):
            param_bounds = [param_bounds] * self.num_params
        elif len(param_bounds) != self.num_params:
            raise ValueError("Dimensions of resolutions and bounds do not match")
        self.param_bounds: NDArray[np.float_] = np.array(param_bounds)
        self.param_grid: NDArray[np.float_] = self._gen_param_grid()
        self.landscape: LandscapeData | None = None
        self.sampled_landscape: NDArray[np.float_] | None = None
        self._sampled_indices: NDArray[np.int_] | None = None
        self._interpolator: RegularGridInterpolator | None = None

    def __call__(self, params: Sequence[float]) -> float:
        return self.interpolator(params)[0]

    @property
    def interpolator(self) -> RegularGridInterpolator:
        if self._interpolator is None:
            warnings.warn(
                "Interpolator not found. "
                "Attempting to initialize the interpolator with default configurations..."
            )
            self.interpolate()
        return self._interpolator

    @property
    def num_samples(self) -> int:
        return 0 if self._sampled_indices is None else len(self._sampled_indices)

    @property
    def optimal_params(self) -> NDArray[np.float_]:
        lower_bounds, upper_bounds = self.param_bounds.T
        return (
            self.optimal_point_indices / self.param_resolutions * (upper_bounds - lower_bounds)
            + lower_bounds
        )

    @property
    def optimal_point_indices(self) -> NDArray[np.int_]:
        return np.array(self._unravel_index(self.landscape.argmin)).flatten()

    @property
    def optimal_value(self) -> float:
        return self.landscape.min

    @property
    def shape(self) -> tuple[int]:
        return tuple(self.param_resolutions)

    @property
    def size(self) -> int:
        return np.prod(self.param_resolutions)

    @property
    def sampled_indices(self) -> Sequence[NDArray[np.int_]] | None:
        if self._sampled_indices is None:
            return None
        return self._unravel_index(self._sampled_indices)

    @property
    def sampling_fraction(self) -> float:
        return self.num_samples / self.size

    def copy(self) -> Landscape:
        return deepcopy(self)

    def interpolate(
        self,
        method: Literal["linear", "nearest", "slinear", "cubic", "quintic", "pchip"] = "slinear",
        bounds_error: bool = False,
        fill_value: float | None = None,
    ) -> RegularGridInterpolator:
        self._interpolator = RegularGridInterpolator(
            self._gen_axes(),
            self.landscape.to_numpy(),  # TODO: implement tensor network version
            method,
            bounds_error,
            fill_value,
        )
        return self._interpolator

    @staticmethod
    def like(landscape: Landscape) -> Landscape:
        return Landscape(landscape.param_resolutions, landscape.param_bounds)

    @staticmethod
    def load(filename: str) -> Landscape:
        return pickle.load(open(filename, "rb"))

    def sample_and_run(
        self,
        executor: BaseExecutor,
        sampling_fraction: float | None = None,
        num_samples: int | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> NDArray[np.float_]:
        if sampling_fraction is None:
            if num_samples is None:
                raise ValueError("Either `sampling_fraction` or `num_samples` is needed.")
        else:
            if num_samples is None:
                num_samples = round(sampling_fraction * self.size)
            else:
                raise ValueError("Supply only one of `sampling_fraction` and `num_samples`.")
        return self.run_flatten_indices(executor, self._sample_indices(num_samples, rng))

    def slice(self, slices: Sequence[slice | int] | slice | int) -> Landscape:
        if isinstance(slices, int) or isinstance(slices, slice):
            slices = [slices]
        slices = tuple(slices) + (slice(None),) * (self.num_params - len(slices))
        axes = self._gen_axes()
        resolutions = [len(axis[s]) for axis, s in zip(axes, slices) if isinstance(s, slice)]
        bounds = [
            (axis[s][0], axis[s][-1]) for axis, s in zip(axes, slices) if isinstance(s, slice)
        ]
        landscape = Landscape(resolutions, bounds)
        landscape.landscape = deepcopy(self.landscape[slices])
        return landscape

    def run_all(self, executor: BaseExecutor) -> NDArray[np.float_]:
        self.landscape = self._run(executor, self._flatten_param_grid()).reshape(
            self.param_resolutions
        )
        return self.landscape

    def run_indices(
        self,
        executor: BaseExecutor,
        param_indices: Sequence[Sequence[int]] | Sequence[int] | int,
    ) -> NDArray[np.float_]:
        if isinstance(param_indices, int):
            param_indices = [param_indices]
        if isinstance(param_indices[0], int):
            param_indices = tuple([i] for i in param_indices)
        param_indices = self._ravel_multi_index(param_indices)
        return self.run_flatten_indices(executor, param_indices)

    def run_flatten_indices(
        self, executor: BaseExecutor, param_indices: Sequence[int] | int
    ) -> NDArray[np.float_]:
        if isinstance(param_indices, int):
            param_indices = [param_indices]
        param_indices = np.array(param_indices)
        result = self._run(
            executor,
            self._flatten_param_grid()[param_indices],
        )
        self._add_sampled_landscape(param_indices, result)
        return result

    def reconstruct(self, reconstructor: BaseReconstructor | None = None) -> NDArray[np.float_]:
        if reconstructor is None:
            reconstructor = BPReconstructor()
        self.landscape = reconstructor.run(self)
        return self.landscape

    def to_dense(self) -> TensorLandscapeData:
        self.landscape = self.landscape.to_dense()
        return self.landscape

    def to_tensor_network(self) -> TensorNetworkLandscapeData:
        self.landscape = self.landscape.to_tensor_network()
        return self.landscape

    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(self, open(filename, "wb"))

    def _add_sampled_landscape(
        self, sampled_indices: NDArray[np.int_], sampled_landscape: NDArray[np.float_]
    ) -> None:
        if self._sampled_indices is not None:
            sampled_indices = np.concatenate((sampled_indices, self._sampled_indices))
            sampled_landscape = np.concatenate((sampled_landscape, self.sampled_landscape))
        sampled_indices, value_indices = np.unique(sampled_indices, True)
        self._sampled_indices = sampled_indices
        self.sampled_landscape = sampled_landscape[value_indices]

    def _flatten_param_grid(self) -> NDArray[np.float_]:
        return self.param_grid.reshape(-1, self.num_params)

    def _gen_axes(self) -> list[NDArray[np.float_]]:
        return [
            np.linspace(
                self.param_bounds[i][0],
                self.param_bounds[i][1],
                self.param_resolutions[i],
            )
            for i in range(self.num_params)
        ]

    def _gen_param_grid(self) -> NDArray[np.float_]:
        return np.array(
            np.meshgrid(
                *self._gen_axes(),
                indexing="ij",
            )
        ).transpose((*range(1, self.num_params + 1), 0))

    def _ravel_multi_index(self, indices: Sequence[NDArray[np.int_]]) -> NDArray[np.int_]:
        return np.ravel_multi_index(indices, self.param_resolutions)

    def _run(
        self, executor: BaseExecutor, params_list: Sequence[Sequence[float]]
    ) -> NDArray[np.float_]:
        return executor.run_batch(params_list)

    def _sample_indices(
        self, num_samples: int, rng: np.random.Generator | int | None = None
    ) -> NDArray[np.int_]:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        # return rng.choice(
        #     np.arange(self.size)[np.in1d(np.arange(self.size), self._sampled_indices, True, True)],
        #     num_samples,
        #     replace=False,
        # )
        import teneva

        return self._ravel_multi_index(
            teneva.sample_lhs(self.param_resolutions, num_samples, rng).T
        )

    def _unravel_index(self, indices: Sequence[int]) -> tuple[NDArray[np.int_]]:
        return np.unravel_index(indices, self.param_resolutions)

    def __getitem__(self, key: Sequence[slice | int] | slice | int) -> Landscape:
        return self.slice(key)
