"""
A collection of descriptive type aliases used across the library.
"""

import os
from typing import Tuple, Union

import shapely

#: Something that can be used as a path.
Pathish = Union[os.PathLike, str]

#: Bounds of a geometry or of an area.
#: Components are ordered as (min x, min y, max x, max y).
Bounds = Tuple[float, float, float, float]

#: A landmark for a plot.
#: This is a tuple of the landmark name and and its location.
Landmark = Tuple[str, shapely.Point]
