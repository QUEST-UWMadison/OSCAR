from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..landscape.landscape import Landscape


class BaseReconstructor(ABC):
    @abstractmethod
    def run(
        self, landscape: Landscape, verbose: bool = False, callback: Callable | None = None
    ) -> NDArray[np.float64]:
        pass
