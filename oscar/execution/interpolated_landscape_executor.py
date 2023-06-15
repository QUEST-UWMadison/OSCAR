from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..landscape import Landscape
from .base_executor import BaseExecutor


class InterpolatedLandscapeExecutor(BaseExecutor):
    def __init__(self, landscape: Landscape) -> None:
        self.landscape: Landscape = landscape

    def _run(self, params: Sequence[float]) -> float:
        return self.landscape.interpolator(params)
