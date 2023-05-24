from __future__ import annotations

import math
from time import time
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Optional

import numpy as np

from ..execution import BaseExecutor

if TYPE_CHECKING:
    from ..reconstruction.base_reconstructor import BaseReconstructor


class Landscape:
    def __init__(
        self,
        param_resolutions: Iterable[int],
        param_bounds: Optional[Iterable[tuple[float, float]] | tuple[float, float]] = (
            -math.pi / 2,
            math.pi / 2,
        ),
    ):
        self.num_params = len(param_resolutions)
        self.param_resolutions = np.array(param_resolutions)
        if not isinstance(param_bounds[0], tuple):
            param_bounds = [param_bounds] * self.num_params
        self.param_bounds = np.array(param_bounds)
        self.param_grid = self._gen_param_grid()
        self.true_landscape = None
        self.reconstructed_landscape = None
        self.sampled_landscape = None
        self._sampled_indices = None

    @property
    def size(self):
        return np.prod(self.param_resolutions)

    @property
    def shape(self):
        return self.param_resolutions.tolist()

    @property
    def sampled_indices(self):
        if self._sampled_indices is None:
            return None
        return self._unravel_index(self._sampled_indices)

    @sampled_indices.setter
    def sampled_indices(self, indices: Iterable[np.ndarray]):
        self._sample_indices = self.ravel_multi_index(indices)

    @property
    def num_samples(self):
        return 0 if self._sampled_indices is None else len(self._sampled_indices)

    @property
    def sampling_fraction(self):
        return self.num_samples / self.size

    def _gen_param_grid(self) -> np.ndarray:
        return np.array(
            np.meshgrid(
                *[
                    np.linspace(
                        self.param_bounds[i][0],
                        self.param_bounds[i][1],
                        self.param_resolutions[i],
                    )
                    for i in range(self.num_params)
                ],
                indexing="ij",
            )
        ).transpose((*range(1, self.num_params + 1), 0))

    def _flatten_param_grid(self) -> np.ndarray:
        return self.param_grid.reshape(-1, self.num_params)

    def _run(self, executor: BaseExecutor, param_list: Iterable[np.ndarray]) -> np.ndarray:
        times, data = [], []
        for params in param_list:
            start_time = time()
            data.append(executor.run(params))
            times.append(time() - start_time)
        return np.array(data)

    def run_all(self, executor: BaseExecutor) -> np.ndarray:
        self.true_landscape = self._run(executor, self._flatten_param_grid()).reshape(
            self.param_resolutions
        )
        return self.true_landscape

    def run_after_sample(
        self,
        executor: BaseExecutor,
        sampling_fraction: Optional[float] = None,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
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

    def run_with_indices(
        self,
        executor: BaseExecutor,
        param_indices: tuple[Iterable[int]] | Iterable[int] | int,
    ) -> np.ndarray:
        if isinstance(param_indices, int):
            param_indices = [param_indices]
        if isinstance(param_indices[0], int):
            param_indices = tuple([i] for i in param_indices)
        param_indices = self._ravel_multi_index(param_indices)
        return self.run_with_flatten_indices(executor, param_indices)

    def run_with_flatten_indices(
        self, executor: BaseExecutor, param_indices: Iterable[int] | int
    ) -> np.ndarray:
        if isinstance(param_indices, int):
            param_indices = [param_indices]
        if self._sampled_indices is None:
            self._sampled_indices = np.array(param_indices)
        else:
            self._sampled_indices = np.unique(
                np.concatenate((self._sampled_indices, param_indices))
            )
        if self.true_landscape is not None:
            return self.true_landscape.flat[param_indices]
        return self._run(
            executor,
            self._flatten_param_grid()[param_indices],
        )

    def reconstruct(self, reconstructor: BaseReconstructor) -> np.ndarray:
        self.reconstructed_landscape = reconstructor.run(self)
        return self.reconstructed_landscape

    def reconstruction_error(
        self,
        metric: Literal["MIN_MAX", "MEAN", "RMSE", "CROSS_CORRELATION", "CONV", "NRMSE", "ZNCC"]
        | Callable[[np.ndarray, np.ndarray], float],
    ) -> float:
        if self.true_landscape is None:
            raise RuntimeError(
                "The true landscape is needed. Use `Landscape.run_all()` or "
                "directly supply it to `Landscape.true_landscape`."
            )
        if self.reconstructed_landscape is None:
            raise RuntimeError(
                "The reconstructed landscape is needed. Use `Landscape.reconstruct()` or "
                "directly supply it to `Landscape.reconstructed_landscape`."
            )
        if isinstance(metric, callable):
            return metric(self.true_landscape, self.reconstructed_landscape)
        else:
            metric = metric.upper()
            raise NotImplementedError()

    def _sample_indices(self, num_samples: int, seed: Optional[int] = None) -> np.ndarray:
        return np.random.default_rng(seed).choice(self.size, num_samples, replace=False)

    def _unravel_index(self, indices: Iterable[int]) -> Iterable[np.ndarray]:
        return np.unravel_index(indices, self.param_resolutions)

    def _ravel_multi_index(self, indices: Iterable[np.ndarray]) -> Iterable[int]:
        return np.ravel_multi_index(indices, self.param_resolutions)

    def _add_sampled_landscape(self):
        raise NotImplementedError()

    def visualize_2d(
        self, which_landscape: Optional[Literal["true", "reconstructed", "both"]] = None
    ):
        raise NotImplementedError()
