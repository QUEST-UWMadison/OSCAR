from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..landscape.landscape import Landscape


class BaseReconstructor(ABC):
    @abstractmethod
    def run(self, lanscape: Landscape) -> NDArray[np.float_]:
        pass
