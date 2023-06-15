from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..landscape.landscape import Landscape


class BaseReconstructor(ABC):
    @abstractmethod
    def run(self, lanscape: Landscape) -> NDArray[np.float_]:
        pass
