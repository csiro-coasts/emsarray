"""
Datasets following the CF conventions with gridded datasets.
Both 1D coordinates and 2D coordinates are supported.
"""
from __future__ import annotations

import abc
import enum
import itertools
import warnings
from contextlib import suppress
from functools import cached_property
from typing import (
    Dict, Generic, Hashable, List, Optional, Sequence, Tuple, Type, TypeVar,
    cast
)

import numpy
import xarray
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from emsarray import masking, utils
from emsarray.exceptions import ConventionViolationWarning
from emsarray.types import Bounds, Pathish

from ._base import DimensionConvention, Specificity


class CFGridKind(str, enum.Enum):
    """"""
    face = 'face'


#: A two-tuple of ``(y, x)``.
CFGridIndex = Tuple[int, int]


CF_LATITUDE_UNITS = {
    'degrees_north', 'degree_north', 'degree_N', 'degrees_N',
    'degreeN', 'degreesN'
}

CF_LONGITUDE_UNITS = {
    'degrees_east', 'degree_east', 'degree_E', 'degrees_E',
    'degreeE', 'degreesE'
}


class CFGridTopology(abc.ABC):
    """
    A topology helper that keeps track of the latitude and longitude coordinates
    in a CF grid dataset.
    """
    def __init__(
        self,
        dataset: xarray.Dataset,
        longitude: Optional[Hashable] = None,
        latitude: Optional[Hashable] = None,
    ):
        """
        Construct a new :class:`CFGridTopology` instance for a dataset.

        By default this will introspect the dataset
        looking for a longitude and latitude coordinate variable.
        The ``longitude`` and ``latitude`` parameters
        allow you to manually specify the correct coordinate names
        if the automatic detection fails.
        """
        self.dataset = dataset
        if longitude is not None:
            self.longitude_name = longitude
        if latitude is not None:
            self.latitude_name = latitude

    @cached_property
    def latitude_name(self) -> Hashable:
        """
        The name of the latitude coordinate variable.
        Found by looking for a variable with either a
        ``standard_name = "latitude"`` or
        ``units = "degree_north"``
        attribute.

        See Also
        --------
        https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#latitude-coordinate
        """
        try:
            return next(
                name
                for name, variable in self.dataset.variables.items()
                if variable.attrs.get('units') in CF_LATITUDE_UNITS
                or variable.attrs.get('standard_name') == 'latitude'
                or variable.attrs.get('axis') == 'Y'
            )
        except StopIteration:
            raise ValueError("Could not find latitude coordinate")

    @cached_property
    def longitude_name(self) -> Hashable:
        """
        The name of the longitude coordinate variable.
        Found by looking for a variable with either a
        ``standard_name = "longitude"`` or
        ``units = "degree_east"``
        attribute.

        See Also
        --------
        https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#longitude-coordinate
        """
        try:
            return next(
                name for name, variable in self.dataset.variables.items()
                if variable.attrs.get('units') in CF_LONGITUDE_UNITS
                or variable.attrs.get('standard_name') == 'longitude'
                or variable.attrs.get('axis') == 'X'
            )
        except StopIteration:
            raise ValueError("Could not find longitude coordinate")

    @property
    def latitude(self) -> xarray.DataArray:
        """The latitude coordinate variable"""
        return self.dataset[self.latitude_name]

    @property
    def longitude(self) -> xarray.DataArray:
        """The longitude coordinate variable"""
        return self.dataset[self.longitude_name]

    @property
    @abc.abstractmethod
    def latitude_bounds(self) -> xarray.DataArray:
        """
        Bounds for the latitude coordinate variable.
        If there are no bounds defined on the dataset,
        bounds will be generated.
        """
        pass

    @property
    @abc.abstractmethod
    def longitude_bounds(self) -> xarray.DataArray:
        """
        Bounds for the longitude coordinate variable.
        If there are no bounds defined on the dataset,
        bounds will be generated.
        """
        pass

    @property
    @abc.abstractmethod
    def y_dimension(self) -> Hashable:
        """
        The name of the 'y' dimension.
        For 1D coordinates, this is the the latitude dimension.
        For 2D coordinates, this is the first dimension of the coordinate variables.
        """
        pass

    @property
    @abc.abstractmethod
    def x_dimension(self) -> Hashable:
        """
        The name of the 'x' dimension.
        For 1D coordinates, this is the the longitude dimension.
        For 2D coordinates, this is the second dimension of the coordinate variables.
        """
        pass

    @cached_property
    def shape(self) -> Tuple[int, int]:
        """The shape of this grid, as a tuple of ``(y, x)``."""
        dims = self.dataset.dims
        return (dims[self.y_dimension], dims[self.x_dimension])

    @cached_property
    def size(self) -> int:
        """
        The scalar size of this grid.

        Equal to ``numpy.prod(topology.shape)``,
        i.e., the product of the grid dimensions.
        """
        return int(numpy.prod(self.shape))


Topology = TypeVar('Topology', bound=CFGridTopology)


class CFGrid(Generic[Topology], DimensionConvention[CFGridKind, CFGridIndex]):
    """
    A base class for CF grid datasets.
    There are two concrete subclasses: :class:`CFGrid1D` and :class:`CFGrid2D`.
    """

    grid_kinds = frozenset(CFGridKind)
    default_grid_kind = CFGridKind.face
    topology_class: Type[Topology]

    def __init__(
        self,
        dataset: xarray.Dataset,
        *,
        latitude: Optional[Hashable] = None,
        longitude: Optional[Hashable] = None,
        topology: Optional[Topology] = None,
    ) -> None:
        """
        Construct a new :class:`CFGrid` instance.

        Parameters
        ----------
        dataset : :class:`xarray.Dataset`
            A :class:`~xarray.Dataset` that follows the CF conventions.
            The grid coordinates must be one-dimensional.
        latitude : Hashable, optional
            The name of the latitude coordinate variable on this dataset. Optional.
            By default the coordinate variables are found by introspecting the dataset.
            You can use this parameter to override this behaviour.
        longitude : Hashable, optional
            The name of the longitude coordinate variable on this dataset. Optional.
            By default the coordinate variables are found by introspecting the dataset.
            You can use this parameter to override this behaviour.
        topology : :class:`CFGridTopology`, optional
            Optional, allows you to override the default topology helper.
        """
        super().__init__(dataset)
        if latitude is not None and longitude is not None:
            if topology is not None:
                raise TypeError(
                    "Can not pass both latitude and longitude arguments, "
                    "and the topology argument")
            topology = self.topology_class(
                self.dataset, latitude=latitude, longitude=longitude)

        if topology is not None:
            self.topology = topology

    @cached_property
    def topology(self) -> Topology:
        """A :class:`CFGridTopology` helper."""
        return self.topology_class(self.dataset)

    @cached_property
    def bounds(self) -> Bounds:
        # This can be computed easily from the coordinate bounds
        topology = self.topology
        min_x = numpy.nanmin(topology.longitude_bounds)
        max_x = numpy.nanmax(topology.longitude_bounds)
        min_y = numpy.nanmin(topology.latitude_bounds)
        max_y = numpy.nanmax(topology.latitude_bounds)
        return (min_x, min_y, max_x, max_y)

    @cached_property
    def grid_dimensions(self) -> Dict[CFGridKind, Sequence[Hashable]]:
        return {
            CFGridKind.face: [self.topology.y_dimension, self.topology.x_dimension],
        }

    def unpack_index(self, index: CFGridIndex) -> Tuple[CFGridKind, Sequence[int]]:
        return CFGridKind.face, index

    def pack_index(self, grid_kind: CFGridKind, indices: Sequence[int]) -> CFGridIndex:
        return cast(CFGridIndex, indices)

    def get_all_geometry_names(self) -> List[Hashable]:
        # Grid datasets contain latitude and longitude variables
        # plus optional bounds variables.
        names = [
            self.topology.longitude_name,
            self.topology.latitude_name,
        ]

        bounds_names: List[Optional[Hashable]] = [
            self.topology.longitude.attrs.get('bounds', None),
            self.topology.latitude.attrs.get('bounds', None),
        ]
        for bounds_name in bounds_names:
            if bounds_name is not None and bounds_name in self.dataset.variables:
                names.append(bounds_name)

        return names

    def drop_geometry(self) -> xarray.Dataset:
        dataset = super().drop_geometry()
        dataset.attrs.pop('Conventions', None)
        return dataset

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xarray.Dataset:
        topology = self.topology

        intersecting_indices = self.strtree.query(clip_geometry, predicate='intersects')
        mask = numpy.full(topology.shape, fill_value=False)
        mask.ravel()[intersecting_indices] = True

        if buffer > 0:
            mask = masking.blur_mask(mask, size=buffer)
        dimensions = [topology.y_dimension, topology.x_dimension]

        return xarray.Dataset(
            data_vars={
                'cell_mask': xarray.DataArray(data=mask, dims=dimensions),
            },
            coords={
                topology.latitude_name: topology.latitude.copy(),
                topology.longitude_name: topology.longitude.copy(),
            },
            attrs={'type': 'CFGrid mask'},
        )

    def apply_clip_mask(self, clip_mask: xarray.Dataset, work_dir: Pathish) -> xarray.Dataset:
        return masking.mask_grid_dataset(self.dataset, clip_mask, work_dir)


# 1D coordinate grids

class CFGrid1DTopology(CFGridTopology):
    """
    Collects information about the topology of a gridded dataset
    so that you don't have to.

    This grid has one-dimensional coordinates,
    such as ``lat(lat)`` or ``longitude(x)``.
    """

    @cached_property
    def y_dimension(self) -> Hashable:
        """The name of the latitude dimension, taken from the latitude coordinate"""
        return self.latitude.dims[0]

    @cached_property
    def x_dimension(self) -> Hashable:
        """The name of the latitude dimension, taken from the latitude coordinate"""
        return self.longitude.dims[0]

    def _get_or_make_bounds(self, coordinate: xarray.DataArray) -> xarray.DataArray:
        with suppress(KeyError):
            bounds = self.dataset.data_vars[coordinate.attrs['bounds']]
            if (
                len(bounds.dims) == 2
                and bounds.dims[0] == coordinate.dims[0]
                and self.dataset.dims[bounds.dims[1]] == 2
            ):
                return bounds
            else:
                expected_dims = (coordinate.dims[0], 2)
                warnings.warn(
                    f"Bounds {bounds.name!r} for coordinate {coordinate.name!r} "
                    f"had invalid dimensions. "
                    f"Expected {expected_dims!r}, "
                    f"found {tuple(bounds.dims)!r}",
                    category=ConventionViolationWarning,
                    stacklevel=4)

        values = coordinate.values
        first_gap = values[1] - values[0]
        last_gap = values[-1] - values[-2]
        mid_points = numpy.concatenate([
            [values[0] - first_gap / 2],
            (values[1:] + values[:-1]) / 2,
            [values[-1] + last_gap / 2]
        ])
        return xarray.DataArray(
            numpy.stack([mid_points[:-1], mid_points[1:]], axis=-1),
            dims=[coordinate.dims[0], 'bounds'],
        )

    @cached_property
    def latitude_bounds(self) -> xarray.DataArray:
        """North/south bounds for each latitude point"""
        return self._get_or_make_bounds(self.latitude)

    @cached_property
    def longitude_bounds(self) -> xarray.DataArray:
        """East/west bounds for each longitude point"""
        return self._get_or_make_bounds(self.longitude)


class CFGrid1D(CFGrid[CFGrid1DTopology]):
    """A :class:`.Convention` subclass representing datasets on an axis-aligned grid
    that follows the CF metadata conventions
    and has one dimensional coordinates.
    """
    topology_class = CFGrid1DTopology

    @classmethod
    def check_dataset(cls, dataset: xarray.Dataset) -> Optional[int]:
        """
        A dataset is a 1D CF grid if it has one dimensional
        latitude and longitude coordinate variables.

        Many other conventions extend the CF conventions,
        so this matches with low specificity.
        """
        topology = cls.topology_class(dataset)
        try:
            latitude = topology.latitude
            longitude = topology.longitude
        except ValueError:
            return None

        # Must have one dimensional coordinates
        if len(latitude.dims) != 1 or len(longitude.dims) != 1:
            return None

        return Specificity.LOW

    @cached_property
    @utils.timed_func
    def polygons(self) -> numpy.ndarray:
        lon_bounds = self.topology.longitude_bounds.values
        lat_bounds = self.topology.latitude_bounds.values

        # Make a bounds array as if this dataset had 2D coordinates.
        # 1D bounds are (min, max).
        # 2D bounds are (j-i, i-1), (j-1, i+1), (j+1, i+1), (j+1, i-1).
        # The bounds values are repeated as required, are given a new dimension,
        # then repeated along that new dimension.
        # They will come out as array with shape (lat, lon, 4)
        y_size, x_size = self.topology.shape
        lon_bounds_2d = numpy.tile(lon_bounds[numpy.newaxis, :, [0, 1, 1, 0]], (y_size, 1, 1))
        lat_bounds_2d = numpy.tile(lat_bounds[:, numpy.newaxis, [0, 0, 1, 1]], (1, x_size, 1))

        # points is a (topology.size, 4, 2) array of the corners of each cell
        points = numpy.stack([lon_bounds_2d, lat_bounds_2d], axis=-1).reshape((-1, 4, 2))

        return utils.make_polygons_with_holes(points)

    @cached_property
    def face_centres(self) -> numpy.ndarray:
        topology = self.topology
        xx, yy = numpy.meshgrid(topology.longitude.values, topology.latitude.values)
        centres = numpy.column_stack((xx.flatten(), yy.flatten()))
        return cast(numpy.ndarray, centres)

    @cached_property
    def geometry(self) -> Polygon:
        # As CFGrid1D is axis aligned,
        # the geometry can be constructed from the bounds.
        return box(*self.bounds)


# 2D coordinate grids

class CFGrid2DTopology(CFGridTopology):
    """
    Collects information about the topology of a gridded dataset
    so that you don't have to.

    This grid has two-dimensional coordinates,
    such as ``lat(y, x)`` and ``lon(y, x)``
    """
    @property
    def y_dimension(self) -> Hashable:
        """
        The name of the first dimension of the coordinate variables.
        This is nominally called the ``y`` dimension.
        """
        return self.latitude.dims[0]

    @property
    def x_dimension(self) -> Hashable:
        """
        The name of the second dimension of the coordinate variables.
        This is nominally called the ``x`` dimension.
        """
        return self.latitude.dims[1]

    def _get_or_make_bounds(self, coordinate: xarray.DataArray) -> xarray.DataArray:
        # Use the bounds defined on the coordinate itself, if any.
        with suppress(KeyError):
            bounds = self.dataset[coordinate.attrs['bounds']]
            if (
                len(bounds.dims) == 3
                and bounds.dims[0] == self.y_dimension
                and bounds.dims[1] == self.x_dimension
                and self.dataset.dims[bounds.dims[2]] == 4
            ):
                return cast(xarray.DataArray, bounds)
            else:
                expected_dims = (self.y_dimension, self.x_dimension, 2)
                warnings.warn(
                    f"Bounds {bounds.name!r} for coordinate {coordinate.name!r} "
                    f"had invalid dimensions. "
                    f"Expected {expected_dims!r}, "
                    f"found {tuple(bounds.dims)!r}",
                    category=ConventionViolationWarning,
                    stacklevel=4)

        # The dataset has no bounds defined. Let us make our own.
        #
        # Assume the bounds for each cell are contiguous with the neighbouring cells.
        # The corners are the average of the four surrounding cell centres.
        # On the edges where there are fewer 'surrounding' cells, the cell centres are used.
        # Edge and corner cells will be smaller than the surrounding cells because of this.

        # numpy.nanmean will return nan for an all-nan column.
        # This is the exact behaviour that we want.
        # numpy emits a warning that can not be silenced when this happens,
        # so that warning is temporarily ignored.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Mean of empty slice", category=RuntimeWarning)
            grid = numpy.nanmean([
                numpy.pad(coordinate.values, pad, constant_values=numpy.nan)
                for pad in itertools.product([(1, 0), (0, 1)], [(1, 0), (0, 1)])
            ], axis=0)

        y_size, x_size = self.shape
        bounds = numpy.array([
            [
                [grid[y, x], grid[y, x + 1], grid[y + 1, x + 1], grid[y + 1, x]]
                for x in range(x_size)
            ]
            for y in range(y_size)
        ])
        # Any cell that has a `nan` in its bounds will be set to all nan
        cells_with_nans = numpy.isnan(bounds).any(axis=2)
        bounds[cells_with_nans] = numpy.nan

        return xarray.DataArray(
            bounds,
            dims=[self.y_dimension, self.x_dimension, 'bounds'],
        )

    @cached_property
    def longitude_bounds(self) -> xarray.DataArray:
        return self._get_or_make_bounds(self.longitude)

    @property
    def latitude_bounds(self) -> xarray.DataArray:
        return self._get_or_make_bounds(self.latitude)


class CFGrid2D(CFGrid[CFGrid2DTopology]):
    """A :class:`.Convention` subclass representing datasets on a curvilinear grid
    that follows the CF metadata conventions
    and has two dimensional coordinates.
    """
    topology_class = CFGrid2DTopology

    @classmethod
    def check_dataset(cls, dataset: xarray.Dataset) -> Optional[int]:
        """
        A dataset is a 2D CF grid if it has two dimensional
        latitude and longitude coordinate variables.

        Many other conventions extend the CF conventions,
        so this matches with low specificity.
        """
        topology = cls.topology_class(dataset)
        try:
            latitude = topology.latitude
            longitude = topology.longitude
        except ValueError:
            return None

        # Must have two dimensional coordinates
        if len(latitude.dims) != 2 or len(longitude.dims) != 2:
            return None

        return Specificity.LOW

    @cached_property
    @utils.timed_func
    def polygons(self) -> numpy.ndarray:
        # Construct polygons from the bounds of the cells
        lon_bounds = self.topology.longitude_bounds.values
        lat_bounds = self.topology.latitude_bounds.values

        # points is a (topology.size, 4, 2) array of the corners of each cell
        points = numpy.stack([lon_bounds, lat_bounds], axis=-1).reshape((-1, 4, 2))

        return utils.make_polygons_with_holes(points)

    @cached_property
    def face_centres(self) -> numpy.ndarray:
        centres = numpy.column_stack((
            self.ravel(self.topology.longitude).values,
            self.ravel(self.topology.latitude).values,
        ))
        return cast(numpy.ndarray, centres)
