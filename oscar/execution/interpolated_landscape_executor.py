import numpy as np

from .base_executor import BaseExecutor
from ..landscape import Landscape


class InterpolatedLandscapeExecutor(BaseExecutor):
    def __init__(self, landscape: Landscape) -> None:
        self.landscape = landscape

    def _run(self, params: np.ndarray) -> float:
        return self.landscape.interpolator(params)
    