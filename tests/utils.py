import abc
import contextlib
import importlib.metadata
import itertools
import warnings
from collections.abc import Hashable
from functools import cached_property
from typing import Any

import numpy
import pytest
import shapely
import xarray
from packaging.requirements import Requirement

from emsarray.conventions.arakawa_c import (
    ArakawaCGridKind, c_mask_from_centres
)


@contextlib.contextmanager
def filter_warning(*args, record: bool = False, **kwargs):
    """
    A shortcut wrapper around warnings.catch_warning()
    and warnings.filterwarnings()
    """
    with warnings.catch_warnings(record=record) as context:
        warnings.filterwarnings(*args, **kwargs)
        yield context


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


def reduce_axes(arr: numpy.ndarray, axes: tuple[bool, ...] | None = None) -> numpy.ndarray:
    """
    Reduce the size of an array by one on an axis-by-axis basis. If an axis is
    reduced, neigbouring values are averaged together

    :param arr: The array to reduce.
    :param axes: A tuple of booleans indicating which axes should be reduced. Optional, defaults to reducing along all axes.
    :returns: A new array with the same number of axes, but one size smaller in each axis that was reduced.
    """
    if axes is None:
        axes = tuple(True for _ in arr.shape)
    axes_slices = [[numpy.s_[+1:], numpy.s_[:-1]] if axis else [numpy.s_[:]] for axis in axes]
    return numpy.mean([arr[tuple(p)] for p in itertools.product(*axes_slices)], axis=0)  # type: ignore


def mask_from_strings(mask_strings: list[str]) -> numpy.ndarray:
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
    return numpy.array([list(map(int, line)) for line in mask_strings]).astype(bool)


class ShocLayerGenerator(abc.ABC):
    def __init__(self, *, k: int):
        self.k_size = k

    @property
    def standard_vars(self) -> dict[Hashable, xarray.DataArray]:
        return {
            "z_grid": xarray.DataArray(
                data=self.z_grid,
                dims=["k_grid"],
                attrs={
                    "units": "metre",
                    "long_name": "Z coordinate at grid layer faces",
                    "coordinate_type": "Z",
                },
            ),
            "z_centre": xarray.DataArray(
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
    def simple_coords(self) -> dict[Hashable, xarray.DataArray]:
        return {
            "zc": xarray.DataArray(
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
    def z_grid(self) -> numpy.ndarray:
        # k=0 is the deepest layer. The highest layer is at 0m
        return (self.k_size - numpy.arange(self.k_size + 1)) * 0.5

    @cached_property
    def z_centre(self) -> numpy.ndarray:
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
        face_mask: numpy.ndarray | None = None,
        include_bounds: bool = False,
    ):
        self.j_size = j
        self.i_size = i
        self.face_mask = face_mask
        self.include_bounds = include_bounds

    @abc.abstractmethod
    def make_x_grid(self, j: numpy.ndarray, i: numpy.ndarray) -> numpy.ndarray:
        pass

    @abc.abstractmethod
    def make_y_grid(self, j: numpy.ndarray, i: numpy.ndarray) -> numpy.ndarray:
        pass

    @cached_property
    def standard_mask(self) -> xarray.Dataset:
        face_mask = self.face_mask
        if face_mask is None:
            face_mask = numpy.full((self.j_size, self.i_size), True)
        return c_mask_from_centres(face_mask, self.dimensions)

    @cached_property
    def simple_mask(self) -> xarray.Dataset:
        face_mask = self.face_mask
        if face_mask is None:
            face_mask = numpy.full((self.j_size, self.i_size), True)
        return xarray.Dataset(data_vars={
            "centre_mask": xarray.DataArray(data=face_mask, dims=["j", "i"])
        })

    @property
    def standard_vars(self) -> dict[Hashable, xarray.DataArray]:
        return {
            "x_grid": xarray.DataArray(
                data=self.x_grid,
                dims=self.dimensions[ArakawaCGridKind.node],
                attrs={
                    "units": "degrees_east",
                    "long_name": "Longitude at grid corners",
                    "coordinate_type": "longitude",
                    "projection": "geographic",
                },
            ).where(self.standard_mask.data_vars['node_mask']),
            "y_grid": xarray.DataArray(
                data=self.y_grid,
                dims=self.dimensions[ArakawaCGridKind.node],
                attrs={
                    "units": "degrees_north",
                    "long_name": "Latitude at grid corners",
                    "coordinate_type": "latitude",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['node_mask']),
            "x_centre": xarray.DataArray(
                data=self.x_centre,
                dims=self.dimensions[ArakawaCGridKind.face],
                attrs={
                    "long_name": "Longitude at cell centre",
                    "coordinate_type": "longitude",
                    "units": "degrees_east",
                    "projection": "geographic",
                },
            ).where(self.standard_mask.data_vars['face_mask']),
            "y_centre": xarray.DataArray(
                data=self.y_centre,
                dims=self.dimensions[ArakawaCGridKind.face],
                attrs={
                    "long_name": "Latitude at cell centre",
                    "coordinate_type": "latitude",
                    "units": "degrees_north",
                    "projection": "geographic",
                },
            ).where(self.standard_mask.data_vars['face_mask']),
            "x_left": xarray.DataArray(
                data=self.x_left,
                dims=self.dimensions[ArakawaCGridKind.left],
                attrs={
                    "long_name": "Longitude at centre of left face",
                    "coordinate_type": "longitude",
                    "units": "degrees_east",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['left_mask']),
            "y_left": xarray.DataArray(
                data=self.y_left,
                dims=self.dimensions[ArakawaCGridKind.left],
                attrs={
                    "long_name": "Latitude at centre of left face",
                    "coordinate_type": "latitude",
                    "units": "degrees_north",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['left_mask']),
            "x_back": xarray.DataArray(
                data=self.x_back,
                dims=self.dimensions[ArakawaCGridKind.back],
                attrs={
                    "long_name": "Longitude at centre of back face",
                    "coordinate_type": "longitude",
                    "units": "degrees_east",
                    "projection": "geographic",
                }
            ).where(self.standard_mask.data_vars['back_mask']),
            "y_back": xarray.DataArray(
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
    def simple_vars(self) -> dict[str, xarray.DataArray]:
        simple_vars = {}
        if self.include_bounds:
            simple_vars.update({
                'longitude_bounds': xarray.DataArray(
                    numpy.stack([
                        self.x_grid[:-1, :-1],
                        self.x_grid[:-1, +1:],
                        self.x_grid[+1:, +1:],
                        self.x_grid[+1:, :-1],
                    ], axis=2),
                    dims=["j", "i", "bounds"],
                ).where(self.simple_mask.data_vars['centre_mask']),
                'latitude_bounds': xarray.DataArray(
                    numpy.stack([
                        self.y_grid[:-1, :-1],
                        self.y_grid[:-1, +1:],
                        self.y_grid[+1:, +1:],
                        self.y_grid[+1:, :-1],
                    ], axis=2),
                    dims=["j", "i", "bounds"],
                ).where(self.simple_mask.data_vars['centre_mask']),
            })
        return simple_vars

    @property
    def simple_coords(self) -> dict[Hashable, xarray.DataArray]:
        return {
            "longitude": xarray.DataArray(
                data=self.x_centre,
                dims=["j", "i"],
                attrs={
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                    "coordinate_type": "longitude",
                    "units": "degrees_east",
                    "projection": "geographic",
                    **(
                        {"bounds": "longitude_bounds"}
                        if self.include_bounds else {}
                    ),
                },
            ).where(self.simple_mask.data_vars['centre_mask']),
            "latitude": xarray.DataArray(
                data=self.y_centre,
                dims=["j", "i"],
                attrs={
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                    "coordinate_type": "latitude",
                    "units": "degrees_north",
                    "projection": "geographic",
                    **(
                        {"bounds": "latitude_bounds"}
                        if self.include_bounds else {}
                    ),
                },
            ).where(self.simple_mask.data_vars['centre_mask']),
        }

    @cached_property
    def x_grid(self) -> numpy.ndarray:
        return numpy.fromfunction(self.make_x_grid, (self.j_size + 1, self.i_size + 1))

    @cached_property
    def y_grid(self):
        return numpy.fromfunction(self.make_y_grid, (self.j_size + 1, self.i_size + 1))

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
    def make_x_grid(self, j: numpy.ndarray, i: numpy.ndarray) -> numpy.ndarray:
        return 0.1 * (i + j)  # type: ignore

    def make_y_grid(self, j: numpy.ndarray, i: numpy.ndarray) -> numpy.ndarray:
        return 0.1 * (self.i_size - i + j)  # type: ignore


class RadialShocGrid(ShocGridGenerator):
    def make_x_grid(self, j: numpy.ndarray, i: numpy.ndarray) -> numpy.ndarray:
        return 0.1 * (5 + j) * numpy.cos(numpy.pi - i * numpy.pi / (self.i_size))  # type: ignore

    def make_y_grid(self, j: numpy.ndarray, i: numpy.ndarray) -> numpy.ndarray:
        return 0.1 * (5 + j) * numpy.sin(numpy.pi - i * numpy.pi / (self.i_size))  # type: ignore


def assert_property_not_cached(
    instance: Any,
    prop_name: str,
    /,
) -> None:
    __tracebackhide__ = True  # noqa
    cls = type(instance)
    prop = getattr(cls, prop_name)
    assert isinstance(prop, cached_property), \
        "{instance!r}.{prop_name} is not a cached_property"

    cache = instance.__dict__
    assert prop.attrname not in cache, \
        f"{instance!r}.{prop_name} was cached!"


def skip_versions(*requirements: str):
    """
    Skips a test function if any of the version specifiers match.
    """
    invalid_versions = []
    for requirement in map(Requirement, requirements):
        assert not requirement.extras
        assert requirement.url is None
        assert requirement.marker is None

        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            # The package is not installed, so an invalid version isn't installed
            continue

        if version in requirement.specifier:
            invalid_versions.append(
                f'{requirement.name}=={version} matches skipped version specifier {requirement}')

    return pytest.mark.skipif(len(invalid_versions) > 0, reason='\n'.join(invalid_versions))


def only_versions(*requirements: str):
    """
    Runs a test function only if all of the version specifiers match.
    """
    invalid_versions = []
    for requirement in map(Requirement, requirements):
        assert not requirement.extras
        assert requirement.url is None
        assert requirement.marker is None

        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            # The package is not installed, so a required version is not installed
            invalid_versions.append(f'{requirement.name} is not installed')
            continue

        if version not in requirement.specifier:
            invalid_versions.append(
                f'{requirement.name}=={version} does not satisfy {requirement}')

    return pytest.mark.skipif(len(invalid_versions) > 0, reason='\n'.join(invalid_versions))
