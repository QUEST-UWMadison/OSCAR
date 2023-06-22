from __future__ import annotations

import math
import os
import pickle
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from ..execution.base_executor import BaseExecutor
from ..reconstruction import BPReconstructor

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
        if not isinstance(param_bounds[0], Sequence):
            param_bounds = [param_bounds] * self.num_params
        self.param_bounds: NDArray[np.float_] = np.array(param_bounds)
        self.param_grid: NDArray[np.float_] = self._gen_param_grid()
        self.true_landscape: NDArray[np.float_] | None = None
        self.reconstructed_landscape: NDArray[np.float_] | None = None
        self.sampled_landscape: NDArray[np.float_] | None = None
        self._sampled_indices: NDArray[np.int_] | None = None
        self._interpolator: RegularGridInterpolator | None = None

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
        lower_bounds, upper_bounds = self.param_bounds.T  # TODO: arbitrary dimension
        return (
            self.optimal_point_indices
            / self.param_resolutions[:, np.newaxis]
            * (upper_bounds - lower_bounds)[:, np.newaxis]
            + lower_bounds[:, np.newaxis]
        )

    @property
    def optimal_point_indices(self) -> NDArray[np.int_]:
        return self._unravel_index([np.argmin(self._get_landscape())])  # TODO: format

    @property
    def optimal_value(self) -> float:
        return np.min(self._get_landscape())

    @property
    def shape(self) -> list:
        return self.param_resolutions.tolist()

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

    def interpolate(
        self,
        method: Literal["linear", "nearest", "slinear", "cubic", "quintic", "pchip"] = "slinear",
        bounds_error: bool = False,
        fill_value: float | None = None,
    ) -> RegularGridInterpolator:
        landscape = self._get_landscape("auto")
        self._interpolator = RegularGridInterpolator(
            self._gen_axes(),
            landscape,
            method,
            bounds_error,
            fill_value,
        )
        return self._interpolator

    def sample_and_run(
        self,
        executor: BaseExecutor,
        sampling_fraction: float | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float_]:
        if sampling_fraction is None:
            if num_samples is None:
                raise ValueError("Either `sampling_fraction` or `num_samples` is needed.")
        else:
            if num_samples is None:
                num_samples = round(sampling_fraction * self.size)
            else:
                raise ValueError("Supply only one of `sampling_fraction` and `num_samples`.")
        self.sampled_landscape = self.run_with_flatten_indices(
            executor, self._sample_indices(num_samples, seed)
        )
        return self.sampled_landscape

    def run_all(self, executor: BaseExecutor) -> NDArray[np.float_]:
        self.true_landscape = self._run(executor, self._flatten_param_grid()).reshape(
            self.param_resolutions
        )
        return self.true_landscape

    def run_with_indices(
        self,
        executor: BaseExecutor,
        param_indices: Sequence[Sequence[int]] | Sequence[int] | int,
    ) -> NDArray[np.float_]:
        if isinstance(param_indices, int):
            param_indices = [param_indices]
        if isinstance(param_indices[0], int):
            param_indices = tuple([i] for i in param_indices)
        param_indices = self._ravel_multi_index(param_indices)
        return self.run_with_flatten_indices(executor, param_indices)

    def run_with_flatten_indices(
        self, executor: BaseExecutor, param_indices: Sequence[int] | int
    ) -> NDArray[np.float_]:
        if isinstance(param_indices, int):
            param_indices = [param_indices]
        # Commented since the current executor may not be the one used to run the true landscape
        # if self.true_landscape is not None:
        #     result = self.true_landscape.flat[param_indices]
        result = self._run(
            executor,
            self._flatten_param_grid()[param_indices],
        )
        self._add_sampled_landscape(param_indices, result)
        return result

    def reconstruct(self, reconstructor: BaseReconstructor | None = None) -> NDArray[np.float_]:
        if reconstructor is None:
            reconstructor = BPReconstructor()
        self.reconstructed_landscape = reconstructor.run(self)
        return self.reconstructed_landscape

    def reconstruction_error(
        self,
        metric: Literal["MIN_MAX", "MEAN", "RMSE", "CROSS_CORRELATION", "CONV", "NRMSE", "ZNCC"]
        | Callable[[NDArray[np.float_], NDArray[np.float_]], float],
    ) -> float:
        true_landscape = self._get_landscape("true")
        reconstructed_landscape = self._get_landscape("reconstructed")
        if isinstance(metric, callable):
            return metric(true_landscape, reconstructed_landscape)
        else:
            metric = metric.upper()
            raise NotImplementedError()

    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(self, open(filename, "wb"))

    def _add_sampled_landscape(
        self, sampled_indices: NDArray[np.int_], sampled_landscape: NDArray[np.float_]
    ) -> None:
        # sampled_indices = np.unique(sampled_indices)
        # if self._sampled_indices is None:
        self._sampled_indices = np.array(sampled_indices)
        self.sampled_landscape = sampled_landscape
        # else:
        #     # TODO
        #     raise NotImplementedError()
        #     mask = np.isin(sampled_indices, self._sampled_indices, True, True)
        #     run_indices = sampled_indices[mask]
        #     self._sampled_indices = np.unique(
        #         np.concatenate((self._sampled_indices, sampled_indices))
        #     )

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

    def _get_landscape(
        self, which_landscape: Literal["true", "reconstructed", "auto"] = "auto"
    ) -> NDArray[np.float_]:
        if which_landscape == "auto":
            if self.true_landscape is not None:
                return self.true_landscape
            which_landscape = "reconstructed"
        if which_landscape == "true":
            if self.true_landscape is None:
                raise RuntimeError(
                    "The true landscape is not present. Use `Landscape.run_all()` "
                    "or directly supply it to `Landscape.true_landscape`."
                )
            return self.true_landscape
        if which_landscape == "reconstructed":
            if self.reconstructed_landscape is None:
                warnings.warn(
                    "The reconstructed landscape is not present. "
                    "Attempting to reconstruct with default configurations..."
                )
                self.reconstruct()
            return self.reconstructed_landscape

    def _ravel_multi_index(self, indices: Sequence[NDArray[np.int_]]) -> NDArray[np.int_]:
        return np.ravel_multi_index(indices, self.param_resolutions)

    def _run(
        self, executor: BaseExecutor, params_list: Sequence[Sequence[float]]
    ) -> NDArray[np.float_]:
        return executor.run_batch(params_list)

    def _sample_indices(self, num_samples: int, seed: int | None = None) -> NDArray[np.int_]:
        return np.random.default_rng(seed).choice(self.size, num_samples, replace=False)

    def _unravel_index(self, indices: Sequence[int]) -> tuple[NDArray[np.int_]]:
        return np.unravel_index(indices, self.param_resolutions)
