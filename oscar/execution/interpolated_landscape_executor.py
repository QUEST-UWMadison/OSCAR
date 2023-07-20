from __future__ import annotations

from collections.abc import Sequence

from ..landscape import Landscape
from .base_executor import BaseExecutor


class InterpolatedLandscapeExecutor(BaseExecutor):
    def __init__(self, landscape: Landscape) -> None:
        self.landscape: Landscape = landscape

    def _run(self, params: Sequence[float], **kwargs) -> float:
        return self.landscape.interpolator(params, **kwargs)[0]
