from __future__ import annotations

from abc import abstractmethod
from functools import lru_cache
from typing import Sequence

import numpy as np
import teneva
from numpy.typing import NDArray

from .utils import complete_slices


class LandscapeData:
    def __init__(self, data: NDArray[np.float64] | Sequence[NDArray[np.float64]]) -> None:
        self.data: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = data

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def min(self) -> float:
        pass

    @abstractmethod
    def max(self) -> float:
        pass

    @abstractmethod
    def argmin(self) -> NDArray[np.int_]:
        pass

    @abstractmethod
    def argmax(self) -> NDArray[np.int_]:
        pass

    @abstractmethod
    def slice(self, slices: Sequence[slice | int] | slice | int) -> LandscapeData | float:
        pass

    @abstractmethod
    def to_dense(self) -> TensorLandscapeData:
        pass

    @abstractmethod
    def to_tensor_network(self) -> TensorNetworkLandscapeData:
        pass

    @abstractmethod
    def to_numpy(self) -> NDArray[np.float64]:
        pass

    def __getitem__(self, key: Sequence[slice | int] | slice | int) -> LandscapeData:
        return self.slice(key)

    def __array__(self) -> NDArray[np.float64]:
        return self.to_numpy()


class TensorLandscapeData(LandscapeData):
    def __init__(self, data: NDArray[np.float64]) -> None:
        super().__init__(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def min(self) -> float:
        return np.min(self.data)

    def max(self) -> float:
        return np.max(self.data)

    def argmin(self) -> NDArray[np.int_]:
        return np.unravel_index(np.argmin(self.data), self.shape)

    def argmax(self) -> NDArray[np.int_]:
        return np.unravel_index(np.argmax(self.data), self.shape)

    def slice(self, slices: Sequence[slice | int] | slice | int) -> TensorLandscapeData | float:
        data = self.data[slices]
        if isinstance(data, NDArray):
            return TensorLandscapeData(data)
        return data

    def to_dense(self) -> TensorLandscapeData:
        return self

    def to_tensor_network(self) -> TensorNetworkLandscapeData:
        raise NotImplementedError()

    def to_numpy(self) -> NDArray[np.float64]:
        return self.data


class TensorNetworkLandscapeData(LandscapeData):
    def __init__(self, data: Sequence[NDArray[np.float64]]) -> None:
        super().__init__(tuple(data))

    @property
    def active_tensors(self) -> tuple[NDArray[np.float64], ...]:
        return tuple(tsr for tsr in self.data if tsr.ndim == 3)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(tsr.shape[1] for tsr in self.active_tensors)

    @property
    def _pseudo_indices_view(self) -> NDArray[np.float64]:
        return [tsr[:, None, :] if tsr.ndim == 2 else tsr for tsr in self.data]

    def min(self, num_candidates: int | None = None) -> float:
        return self.compute_minima(num_candidates)[1]

    def max(self, num_candidates: int | None = None) -> float:
        return self.compute_minima(num_candidates)[3]

    def argmin(self, num_candidates: int | None = None) -> NDArray[np.int_]:
        return self.compute_minima(num_candidates)[0]

    def argmax(self, num_candidates: int | None = None) -> NDArray[np.int_]:
        return self.compute_minima(num_candidates)[2]

    @lru_cache
    def compute_minima(
        self, num_candidates: int | None = None
    ) -> tuple[NDArray[np.int_], float, NDArray[np.int_], float]:
        if num_candidates is None:
            num_candidates = min(tsr.shape[1] for tsr in self.active_tensors)
        return teneva.optima_tt(self._pseudo_indices_view, num_candidates)

    def slice(
        self, slices: Sequence[slice | int] | slice | int
    ) -> TensorNetworkLandscapeData | float:
        if isinstance(slices, int) or isinstance(slices, slice):
            slices = [slices]
        slices = complete_slices(slices, len(self.data))
        if all(isinstance(s, int) for s in slices):
            return teneva.get(self._pseudo_indices_view, slices)
        tensors, i = [], 0
        for tsr in self.data:
            if tsr.ndim == 2:
                tensors.append(tsr)
            else:
                tensors.append(tsr[:, slices[i]])
                i += 1
        return TensorNetworkLandscapeData(tensors)

    def to_dense(self) -> TensorLandscapeData:
        return TensorLandscapeData(self.to_numpy())

    def to_tensor_network(self) -> TensorNetworkLandscapeData:
        return self

    def to_numpy(self) -> NDArray[np.float64]:
        return teneva.full(self._pseudo_indices_view).reshape(self.shape)
