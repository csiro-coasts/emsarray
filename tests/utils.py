from __future__ import annotations

import abc
import itertools
from functools import cached_property
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
import shapely
import xarray as xr

from emsarray.conventions.arakawa_c import (
    ArakawaCGridKind, c_mask_from_centres
)


def box(minx, miny, maxx, maxy) -> shapely.Polygon:
    """
    Make a box, with coordinates going counterclockwise
    starting at (minx miny).
    """
    return shapely.Polygon([
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
    ])


def reduce_axes(arr: np.ndarray, axes: Optional[Tuple[bool, ...]] = None) -> np.ndarray:
    """
    Reduce the size of an array by one on an axis-by-axis basis. If an axis is
    reduced, neigbouring values are averaged together

    :param arr: The array to reduce.
    :param axes: A tuple of booleans indicating which axes should be reduced. Optional, defaults to reducing along all axes.
    :returns: A new array with the same number of axes, but one size smaller in each axis that was reduced.
    """
    if axes is None:
        axes = tuple(True for _ in arr.shape)
    axes_slices = [[np.s_[+1:], np.s_[:-1]] if axis else [np.s_[:]] for axis in axes]
    return np.mean([arr[tuple(p)] for p in itertools.product(*axes_slices)], axis=0)  # type: ignore


def mask_from_strings(mask_strings: List[str]) -> np.ndarray:
    """
    Make a boolean mask array from a list of strings:

        >>> mask_from_strings([
        ...     "101",
        ...     "010",
        ...     "111",
        ... ])
        array([[ True, False,  True],
               [False,  True, False],
               [ True,  True,  True]])
    """
    return np.array([list(map(int, line)) for line in mask_strings]).astype(bool)


class ShocLayerGenerator(abc.ABC):
    def __init__(self, *, k: int):
        self.k_size = k

    @property
    def standard_vars(self) -> Dict[Hashable, xr.DataArray]:
        return {
            "z_grid": xr.DataArray(
                data=self.z_grid,
                dims=["k_grid"],
                attrs={
                    "units": "metre",
                    "long_name": "Z coordinate at grid layer faces",
                    "coordinate_type": "Z",
                },
            ),
            "z_centre": xr.DataArray(
                data=self.z_centre,
                dims=["k_centre"],
                attrs={
                    "units": "metre",
                    "long_name": "Z coordinate at grid layer centre",
                    "coordinate_type": "Z",
                },
            ),
        }

    @property
    def simple_vars(self) -> Dict[Hashable, xr.DataArray]:
        return {
            "zc": xr.DataArray(
                data=self.z_centre,
                dims=["k"],
                attrs={
                    "units": "metre",
                    "positive": "up",
                    "long_name": "Z coordinate",
                    "axis": "Z",
                    "coordinate_type": "Z",
                },
            ),
        }

    @cached_property
    def z_grid(self) -> np.ndarray:
        # k=0 is the deepest layer. The highest layer is at 0m
        return (self.k_size - np.arange(self.k_size + 1)) * 0.5

    @cached_property
    def z_centre(self) -> np.ndarray:
        return reduce_axes(self.z_grid)


class ShocGridGenerator(abc.ABC):
    dimensions = {
        ArakawaCGridKind.face: ('j_centre', 'i_centre'),
        ArakawaCGridKind.back: ('j_back', 'i_back'),
        ArakawaCGridKind.left: ('j_left', 'i_left'),
        ArakawaCGridKind.node: ('j_node', 'i_node'),
    }

    def __init__(
        self, *,
        j: int,
        i: int,
        face_mask: Optional[np.ndarray] = None
    ):
        self.j_size = j
        self.i_size = i
        self.face_mask = face_mask

    @abc.abstractmethod
    def make_x_grid(self, j: np.ndarray, i: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def make_y_grid(self, j: np.ndarray, i: np.ndarray) -> np.ndarray:
        pass

    @cached_property
    def standard_mask(self) -> xr.Dataset:
        face_mask = self.face_mask
        if face_mask is None:
            face_mask = np.full((self.j_size, self.i_size), True)
        return c_mask_from_centres(face_mask, self.dimensions)

    @cached_property
    def simple_mask(self) -> xr.Dataset:
        face_mask = self.face_mask
        if face_mask is None:
            face_mask = np.full((self.j_size, self.i_size), True)
        return xr.Dataset(data_vars={
            "centre_mask": xr.DataArray(data=face_mask, dims=["j", "i"])
        })

    @property
    def standard_vars(self) -> Dict[Hashable, xr.DataArray]:
        print("Standard mask:")
        print(self.standard_mask)
        return {
            "x_grid": xr.DataArray(
                data=self.x_grid,
                dims=self.dimensions[ArakawaCGridKind.node],
                attrs={
                    "units": "degrees_east",
                    "long_name": "Longitude at grid corners",
                    "coordinate_type": "longitude",
                    "projection": "geographic",
                },
            ).where(self.standard_mask.data_vars['node_mask']),
            "y_grid": xr.DataArray(
                data=self.y_grid,
                dims=self.dimensions[ArakawaCGridKind.node],
                attrs={
                    "units": "degrees_north",
                    "long_name": "Latitude at grid corners",
                    "coordinate_type": "latitude",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['node_mask']),
            "x_centre": xr.DataArray(
                data=self.x_centre,
                dims=self.dimensions[ArakawaCGridKind.face],
                attrs={
                    "long_name": "Longitude at cell centre",
                    "coordinate_type": "longitude",
                    "units": "degrees_east",
                    "projection": "geographic",
                },
            ).where(self.standard_mask.data_vars['face_mask']),
            "y_centre": xr.DataArray(
                data=self.y_centre,
                dims=self.dimensions[ArakawaCGridKind.face],
                attrs={
                    "long_name": "Latitude at cell centre",
                    "coordinate_type": "latitude",
                    "units": "degrees_north",
                    "projection": "geographic",
                },
            ).where(self.standard_mask.data_vars['face_mask']),
            "x_left": xr.DataArray(
                data=self.x_left,
                dims=self.dimensions[ArakawaCGridKind.left],
                attrs={
                    "long_name": "Longitude at centre of left face",
                    "coordinate_type": "longitude",
                    "units": "degrees_east",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['left_mask']),
            "y_left": xr.DataArray(
                data=self.y_left,
                dims=self.dimensions[ArakawaCGridKind.left],
                attrs={
                    "long_name": "Latitude at centre of left face",
                    "coordinate_type": "latitude",
                    "units": "degrees_north",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['left_mask']),
            "x_back": xr.DataArray(
                data=self.x_back,
                dims=self.dimensions[ArakawaCGridKind.back],
                attrs={
                    "long_name": "Longitude at centre of back face",
                    "coordinate_type": "longitude",
                    "units": "degrees_east",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['back_mask']),
            "y_back": xr.DataArray(
                data=self.y_back,
                dims=self.dimensions[ArakawaCGridKind.back],
                attrs={
                    "long_name": "Latitude at centre of back face",
                    "coordinate_type": "latitude",
                    "units": "degrees_north",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['back_mask']),
        }

    @property
    def simple_vars(self) -> Dict[Hashable, xr.DataArray]:
        return {
            "longitude": xr.DataArray(
                data=self.x_centre,
                dims=["j", "i"],
                attrs={
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                    "coordinate_type": "longitude",
                    "units": "degrees_east",
                    "projection": "geographic",
                },
            ).where(self.simple_mask.data_vars['centre_mask']),
            "latitude": xr.DataArray(
                data=self.y_centre,
                dims=["j", "i"],
                attrs={
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                    "coordinate_type": "latitude",
                    "units": "degrees_north",
                    "projection": "geographic",
                },
            ).where(self.simple_mask.data_vars['centre_mask']),
        }

    @cached_property
    def x_grid(self) -> np.ndarray:
        return np.fromfunction(self.make_x_grid, (self.j_size + 1, self.i_size + 1))

    @cached_property
    def y_grid(self):
        return np.fromfunction(self.make_y_grid, (self.j_size + 1, self.i_size + 1))

    @cached_property
    def x_centre(self):
        return reduce_axes(self.x_grid)

    @cached_property
    def y_centre(self):
        return reduce_axes(self.y_grid)

    @cached_property
    def x_left(self):
        return reduce_axes(self.x_grid, (True, False))

    @cached_property
    def y_left(self):
        return reduce_axes(self.y_grid, (True, False))

    @cached_property
    def x_back(self):
        return reduce_axes(self.x_grid, (False, True))

    @cached_property
    def y_back(self):
        return reduce_axes(self.y_grid, (False, True))


class DiagonalShocGrid(ShocGridGenerator):
    def make_x_grid(self, j: np.ndarray, i: np.ndarray) -> np.ndarray:
        return 0.1 * (i + j)  # type: ignore

    def make_y_grid(self, j: np.ndarray, i: np.ndarray) -> np.ndarray:
        return 0.1 * (self.i_size - i + j)  # type: ignore


class RadialShocGrid(ShocGridGenerator):
    def make_x_grid(self, j: np.ndarray, i: np.ndarray) -> np.ndarray:
        return 0.1 * (5 + j) * np.cos(np.pi - i * np.pi / (self.i_size))  # type: ignore

    def make_y_grid(self, j: np.ndarray, i: np.ndarray) -> np.ndarray:
        return 0.1 * (5 + j) * np.sin(np.pi - i * np.pi / (self.i_size))  # type: ignore
