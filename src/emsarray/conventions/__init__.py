"""
:mod:`emsarray` supports multiple data conventions.
Each supported convention :ref:`implements the Convention interface <interface>`
to provide a common base set of functionality.

Calling :func:`.get_dataset_convention` with a dataset will inspect the dataset
and return the most appropriate :class:`Convention` implementation to use.
This can also be accessed using the :ref:`accessor`.

In some cases, the automatic :ref:`accessor`
will not be able to guess the file convention.
In other cases, you might want to override some default parameters.
Convention instances can be instantiated directly in this case.
Refer to each Convention implementation for details.
"""
from ._base import (
    Convention, DimensionConvention, GridKind, Index, SpatialIndexItem,
    Specificity
)
from ._registry import get_dataset_convention, register_convention
from ._utils import open_dataset
from .arakawa_c import ArakawaC
from .grid import CFGrid1D, CFGrid2D
from .shoc import ShocSimple, ShocStandard
from .ugrid import UGrid

__all__ = [
    "Convention", "DimensionConvention", "GridKind", "Index",
    "SpatialIndexItem", "Specificity",
    "get_dataset_convention", "register_convention",
    "open_dataset",
    "ArakawaC",
    "CFGrid1D", "CFGrid2D",
    "ShocSimple", "ShocStandard",
    "UGrid",
]
