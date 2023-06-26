"""Collection of type aliases used across the library."""

import os
from typing import Tuple, Union

Pathish = Union[os.PathLike, str]
Bounds = Tuple[float, float, float, float]
