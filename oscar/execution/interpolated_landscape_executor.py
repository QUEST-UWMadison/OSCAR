from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .base_executor import BaseExecutor

if TYPE_CHECKING:
    from ..landscape import Landscape


class InterpolatedLandscapeExecutor(BaseExecutor):
    def __init__(self, landscape: Landscape) -> None:
        self.landscape: Landscape = landscape

    def _run(self, params: Sequence[float], **kwargs) -> float:
        return self.landscape.interpolator(params, **kwargs)[0]
