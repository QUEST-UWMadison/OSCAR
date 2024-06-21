
from typing import Sequence


def complete_slices(slices: Sequence[slice | int] | slice | int, ndim: int) -> Sequence[slice | int]:
    if isinstance(slices, int) or isinstance(slices, slice):
        slices = [slices]
    return tuple(slices) + (slice(None),) * (ndim - len(slices))
