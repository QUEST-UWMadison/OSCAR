from __future__ import annotations

import math
import os
import pickle
import warnings
from collections.abc import Callable, Generator, Sequence
from copy import deepcopy
from functools import cached_property
from itertools import product
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
import teneva
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from ..execution.base_executor import BaseExecutor
from ..reconstruction.compressed_sensing_reconstructor import CSReconstructor
from .landscape_data import (
    LandscapeData,
    TensorLandscapeData,
    TensorNetworkLandscapeData,
)
from .utils import complete_slices

if TYPE_CHECKING:
    from ..reconstruction.base_reconstructor import BaseReconstructor


class Landscape:
    def __init__(
        self,
        param_resolutions: Sequence[int],
        param_bounds: Sequence[tuple[float, float]] | tuple[float, float] = (
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
        self.param_bounds: NDArray[np.float64] = np.array(param_bounds)
        self.landscape: LandscapeData | None = None
        self.sampled_landscape: NDArray[np.float64] | None = None
        self.sampled_indices: NDArray[np.int_] | None = None
        self._interpolator: RegularGridInterpolator | None = None

    def __call__(self, params: Sequence[float]) -> float:
        return self.interpolator(params)[0]

    @cached_property
    def axes(self) -> tuple[NDArray[np.float64], ...]:
        return tuple(
            np.linspace(
                self.param_bounds[i][0],
                self.param_bounds[i][1],
                self.param_resolutions[i],
            )
            for i in range(self.num_params)
        )

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
        return 0 if self.sampled_indices is None else len(self.sampled_indices)

    @property
    def optimal_params(self) -> NDArray[np.float64]:
        return self.index_to_param(self.optimal_point_index)

    @property
    def optimal_point_index(self) -> NDArray[np.int_]:
        return self.argmin()

    @property
    def optimal_value(self) -> float:
        return self.min()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.param_resolutions)

    @property
    def size(self) -> int:
        return np.prod(self.param_resolutions)

    @property
    def sampled_indices_unravelled(self) -> Sequence[NDArray[np.int_]] | None:
        if self.sampled_indices is None:
            return None
        return self.unravel_index(self.sampled_indices)

    @property
    def sampling_fraction(self) -> float:
        return self.num_samples / self.size

    def argmax(self, *args, **kwargs) -> NDArray[np.int_]:
        return self.landscape.argmax(*args, **kwargs)

    def argmin(self, *args, **kwargs) -> NDArray[np.int_]:
        return self.landscape.argmin(*args, **kwargs)

    def max(self, *args, **kwargs) -> float:
        return self.landscape.max(*args, **kwargs)

    def min(self, *args, **kwargs) -> float:
        return self.landscape.min(*args, **kwargs)

    def copy(self) -> Landscape:
        return deepcopy(self)

    def index_to_param(self, index: Sequence[Sequence[np.int_]]) -> NDArray[np.float64]:
        return np.array([axis[i] for i, axis in zip(index, self.axes)])

    def interpolate(
        self,
        method: Literal["linear", "nearest", "slinear", "cubic", "quintic", "pchip"] = "slinear",
        bounds_error: bool = False,
        fill_value: float = np.nan,
    ) -> RegularGridInterpolator:
        self._interpolator = RegularGridInterpolator(
            self.axes,
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
        executor: BaseExecutor | Landscape,
        sampling_fraction: float | None = None,
        num_samples: int | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> NDArray[np.float64]:
        if sampling_fraction is None:
            if num_samples is None:
                raise ValueError("Either `sampling_fraction` or `num_samples` is needed.")
        else:
            if num_samples is None:
                num_samples = round(sampling_fraction * self.size)
            else:
                raise ValueError("Supply only one of `sampling_fraction` and `num_samples`.")
        return self.run_index(executor, self._sample_indices(num_samples, rng))

    def slice(self, slices: Sequence[slice | int] | slice | int) -> Landscape | float:
        slices = complete_slices(slices, self.num_params)
        data = self.landscape[slices]
        if not isinstance(data, LandscapeData):
            return data
        landscape = Landscape(
            [len(axis[s]) for axis, s in zip(self.axes, slices) if isinstance(s, slice)],
            [
                (axis[s][0], axis[s][-1])
                for axis, s in zip(self.axes, slices)
                if isinstance(s, slice)
            ],
        )
        landscape.landscape = data
        return landscape

    def ravel_multi_index(self, index: Sequence[NDArray[np.int_]]) -> NDArray[np.int_]:
        return np.ravel_multi_index(index, self.param_resolutions)

    def run_all(self, executor: BaseExecutor) -> NDArray[np.float64]:
        self.landscape = TensorLandscapeData(
            self._run(executor, self._gen_params(), self.size).reshape(self.param_resolutions)
        )
        return self.landscape

    def run_index(
        self,
        executor: BaseExecutor | Landscape,
        param_index: Sequence[Sequence[int]] | Sequence[int] | int,
    ) -> NDArray[np.float64]:
        if isinstance(param_index, int):
            param_index = (param_index,)
        if isinstance(param_index[0], int):
            if len(param_index) == self.num_params:
                param_index = (param_index,)
            else:
                param_index = tuple((i,) for i in param_index)
        if isinstance(executor, Landscape):
            if not np.allclose(executor.param_resolutions, self.param_resolutions):
                raise ValueError("Current landscape and source landscape have different resolutions.")
            if not np.allclose(executor.param_bounds, self.param_bounds):
                warnings.warn("Current landscape and source landscape have different parameter bounds.")
            result = np.asarray([executor.slice(idx) for idx in param_index])
        else:
            params = np.asarray(
                [axis[np.array(param_index)[:, i]] for i, axis in enumerate(self.axes)]
            ).T
            result = self._run(executor, params, params.shape[0])
        self._add_sampled_landscape(self.ravel_multi_index(param_index.T), result)
        return result

    def run_flatten_index(
        self, executor: BaseExecutor | Landscape, param_index: Sequence[int] | int
    ) -> NDArray[np.float64]:
        if isinstance(param_index, int):
            param_index = (param_index,)
        return self.run_index(executor, self.unravel_index(param_index))

    def reconstruct(
        self,
        reconstructor: BaseReconstructor | None = None,
        verbose: bool = False,
        callback: Callable | None = None,
    ) -> LandscapeData:
        if reconstructor is None:
            reconstructor = CSReconstructor()
        self.landscape = reconstructor.run(self, verbose, callback)
        return self.landscape

    def to_dense(self) -> Landscape:
        if isinstance(self.landscape, TensorLandscapeData):
            return self
        landscape = Landscape.like(self)
        landscape.landscape = self.landscape.to_dense()
        return landscape

    def to_tensor_network(self) -> Landscape:
        if isinstance(self.landscape, TensorNetworkLandscapeData):
            return self
        landscape = Landscape.like(self)
        landscape.landscape = self.landscape.to_tensor_network()
        return landscape

    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(self, open(filename, "wb"))

    def unravel_index(self, index: Sequence[int]) -> tuple[NDArray[np.int_], ...]:
        return np.unravel_index(index, self.param_resolutions)

    def _add_sampled_landscape(
        self, sampled_indices: NDArray[np.int_], sampled_landscape: NDArray[np.float64]
    ) -> None:
        if self.sampled_indices is not None:
            sampled_indices = np.concatenate((sampled_indices, self.sampled_indices))
            sampled_landscape = np.concatenate((sampled_landscape, self.sampled_landscape))
        sampled_indices, value_indices = np.unique(sampled_indices, True)
        self.sampled_indices = sampled_indices
        self.sampled_landscape = sampled_landscape[value_indices]

    def _gen_params(self) -> Generator[tuple[float, ...], None, None]:
        for params in product(*self.axes):
            yield params

    def _run(
        self, executor: BaseExecutor, params_list: Iterable[Sequence[float]], count: int
    ) -> NDArray[np.float64]:
        return np.fromiter(executor.run_batch(params_list), float, count) # TODO: dtype

    def _sample_indices(
        self, num_samples: int, rng: np.random.Generator | int | None = None
    ) -> NDArray[np.int_]:
        return teneva.sample_lhs(self.param_resolutions, num_samples, rng)

    def __getitem__(self, key: Sequence[slice | int] | slice | int) -> Landscape | float:
        return self.slice(key)
