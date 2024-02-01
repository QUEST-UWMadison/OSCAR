from __future__ import annotations

from abc import abstractmethod
from ast import Not
from typing import Sequence

import numpy as np
import teneva
from numpy.typing import NDArray


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
    def slice(self, slices: Sequence[slice | int] | slice | int) -> LandscapeData:
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

    def slice(self, slices: Sequence[slice | int] | slice | int) -> TensorLandscapeData:
        return TensorLandscapeData(self.data[slices])

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

        def slice(self, slices: Sequence[slice | int] | slice | int) -> TensorNetworkLandscapeData:
            if isinstance(slices, int) or isinstance(slices, slice):
                slices = [slices]
            slices = tuple(slices) + (slice(None),) * (len(self.data) - len(slices))
            # TODO: handle int slices
            return TensorNetworkLandscapeData([tsr[:, slices[i]] for i, tsr in enumerate(self.data) if tsr.shape[1] != 1])

        def to_dense(self) -> TensorLandscapeData:
            return TensorLandscapeData(self.to_numpy())

        def to_tensor_network(self) -> TensorNetworkLandscapeData:
            return self
    
        def to_numpy(self) -> NDArray[np.float_]:
            return teneva.full(self.data)
