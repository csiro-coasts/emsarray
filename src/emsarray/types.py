"""
A collection of descriptive type aliases used across the library.
"""

import os
from collections.abc import Hashable
from typing import Union

import shapely
import xarray

#: Something that can be used as a path.
Pathish = Union[os.PathLike, str]

#: Bounds of a geometry or of an area.
#: Components are ordered as (min x, min y, max x, max y).
Bounds = tuple[float, float, float, float]

#: A landmark for a plot.
#: This is a tuple of the landmark name and and its location.
Landmark = tuple[str, shapely.Point]

#: Either an :class:`xarray.DataArray`, or the name of a data array in a dataset.
DataArrayOrName = Union[Hashable, xarray.DataArray]
