from __future__ import annotations

from abc import abstractmethod
from ast import Not
from typing import Sequence

import numpy as np
import teneva
from numpy.typing import NDArray

from .utils import complete_slices


class LandscapeData:
    def __init__(self, data: NDArray[np.float_] | list[NDArray[np.float_]]) -> None:
        self.data: NDArray[np.float_] | list[NDArray[np.float_]] = data

    @property
    @abstractmethod
    def min(self) -> float:
        pass

    @property
    @abstractmethod
    def max(self) -> float:
        pass

    @property
    @abstractmethod
    def argmin(self) -> int:
        pass

    @property
    @abstractmethod
    def argmax(self) -> int:
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
    def to_numpy(self) -> NDArray[np.float_]:
        pass

    def __getitem__(self, key: Sequence[slice | int] | slice | int) -> LandscapeData:
        return self.slice(key)

    def __array__(self) -> NDArray[np.float_]:
        return self.to_numpy()


class TensorLandscapeData(LandscapeData):
    def __init__(self, data: NDArray[np.float_]) -> None:
        super().__init__(data)

    @property
    def min(self) -> float:
        return np.min(self.data)

    @property
    def max(self) -> float:
        return np.max(self.data)

    @property
    def argmin(self) -> int:
        return np.argmin(self.data)

    @property
    def argmax(self) -> int:
        return np.argmax(self.data)

    def slice(self, slices: Sequence[slice | int] | slice | int) -> TensorLandscapeData | float:
        data = self.data[slices]
        if isinstance(data, NDArray):
            return TensorLandscapeData(data)
        return data

    def to_dense(self) -> TensorLandscapeData:
        return self

    def to_tensor_network(self) -> TensorNetworkLandscapeData:
        raise NotImplementedError()

    def to_numpy(self) -> NDArray[np.float_]:
        return self.data

class TensorNetworkLandscapeData(LandscapeData):
    def __init__(self, data: list[NDArray[np.float_]]) -> None:
        super().__init__(data)

    @property
    def min(self) -> float:
        raise NotImplementedError()

    @property
    def max(self) -> float:
        raise NotImplementedError()

    @property
    def argmin(self) -> int:
        raise NotImplementedError()

    @property
    def argmax(self) -> int:
        raise NotImplementedError()
    
    @property
    def _pseudo_indices_view(self) -> NDArray[np.float_]:
        return [tsr[:, None, :] for tsr in self.data if tsr.ndim == 2]

    def slice(self, slices: Sequence[slice | int] | slice | int) -> TensorNetworkLandscapeData | float:
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

    def to_numpy(self) -> NDArray[np.float_]:
        return teneva.full(self._pseudo_indices_view)
