"""
:mod:`emsarray` supports multiple data formats.
Each supported format :ref:`implements the Format interface <interface>`
to provide a common base set of functionality.

Calling :func:`.get_file_format` with a dataset will inspect the dataset
and return the most appropriate :class:`Format` implementation to use.
This can also be accessed using the :ref:`accessor`.

In some cases, the automatic :ref:`accessor`
will not be able to guess the file format.
In other cases, you might want to override some default parameters.
Format instances can be instantiated directly in this case.
Refer to each Format implementation for details.
"""
from ._base import Format, GridKind, Index, SpatialIndexItem
from ._helpers import (
    Specificity, get_file_format, open_dataset, register_format
)
from .arakawa_c import ArakawaC
from .grid import CFGrid1D, CFGrid2D
from .shoc import ShocSimple, ShocStandard
from .ugrid import UGrid

__all__ = [
    "Format", "GridKind", "Index", "SpatialIndexItem", "Specificity",
    "get_file_format", "open_dataset", "register_format",
    "ArakawaC",
    "CFGrid1D", "CFGrid2D",
    "ShocStandard", "ShocSimple",
    "UGrid",
]
