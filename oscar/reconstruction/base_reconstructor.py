from abc import ABC, abstractmethod

import numpy as np

from ..landscape.landscape import Landscape


class BaseReconstructor(ABC):
    @abstractmethod
    def run(self, lanscape: Landscape) -> np.ndarray:
        pass
