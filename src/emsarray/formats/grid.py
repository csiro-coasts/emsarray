"""
Datasets following the CF conventions with gridded datasets.
Both 1D coordinates and 2D coordinates are supported.
"""
from __future__ import annotations

import abc
import enum
import itertools
import logging
import warnings
from contextlib import suppress
from functools import cached_property
from typing import (
    Dict, Generic, Hashable, Optional, Tuple, Type, TypeVar, cast
)

import numpy as np
import xarray as xr
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon, orient

from emsarray import masking, utils
from emsarray.types import Pathish

from ._base import Format
from ._helpers import Specificity

logger = logging.getLogger(__name__)


class CFGridKind(str, enum.Enum):
    """"""
    face = 'face'


#: A two-tuple of ``(y, x)``.
CFGridIndex = Tuple[int, int]


class CFGridTopology(abc.ABC):
    """
    A topology helper that keeps track of the latitude and longitude coordinates
    in a CF grid dataset.
    """
    def __init__(
        self,
        dataset: xr.Dataset,
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
        """
        try:
            return next(
                name
                for name, variable in self.dataset.variables.items()
                if variable.attrs.get('standard_name') == 'latitude'
                or variable.attrs.get('coordinate_type') == 'latitude'
                or variable.attrs.get('units') == 'degree_north'
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
        """
        try:
            return next(
                name for name, variable in self.dataset.variables.items()
                if variable.attrs.get('standard_name') == 'longitude'
                or variable.attrs.get('coordinate_type') == 'longitude'
                or variable.attrs.get('units') == 'degree_east'
            )
        except StopIteration:
            raise ValueError("Could not find longitude coordinate")

    @property
    def latitude(self) -> xr.DataArray:
        """The latitude coordinate variable"""
        return self.dataset[self.latitude_name]

    @property
    def longitude(self) -> xr.DataArray:
        """The longitude coordinate variable"""
        return self.dataset[self.longitude_name]

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

        Equal to ``np.prod(topology.shape)``,
        i.e., the product of the grid dimensions.
        """
        return int(np.prod(self.shape))


Topology = TypeVar('Topology', bound=CFGridTopology)


class CFGrid(Generic[Topology], Format[CFGridKind, CFGridIndex]):
    """
    A base class for CF grid datasets.
    There are two concrete subclasses: :class:`CFGrid1D` and :class:`CFGrid2D`.
    """

    grid_kinds = frozenset(CFGridKind)
    default_grid_kind = CFGridKind.face
    topology_class: Type[Topology]

    def __init__(
        self,
        dataset: xr.Dataset,
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

    def unravel_index(
        self,
        index: int,
        grid_kind: Optional[CFGridKind] = None,
    ) -> CFGridIndex:
        y, x = map(int, np.unravel_index(index, self.topology.shape))
        return (y, x)

    def ravel_index(self, indices: CFGridIndex) -> int:
        return int(np.ravel_multi_index(indices, self.topology.shape))

    def get_grid_kind_and_size(self, data_array: xr.DataArray) -> Tuple[CFGridKind, int]:
        expected = {self.topology.y_dimension, self.topology.x_dimension}
        dims = set(data_array.dims)
        if dims.issuperset(expected):
            return (CFGridKind.face, self.topology.size)

        expected_sorted = sorted(expected, key=str)
        raise ValueError(f"Data array did not have dimensions {expected_sorted!r}")

    def selector_for_index(self, index: CFGridIndex) -> Dict[Hashable, int]:
        y, x = index
        return {self.topology.y_dimension: y, self.topology.x_dimension: x}

    def drop_geometry(self) -> xr.Dataset:
        dataset = self.dataset.drop_vars([
            self.topology.longitude_name,
            self.topology.latitude_name,
        ])
        dataset.attrs.pop('Conventions', None)

        return dataset

    def make_linear(self, data_array: xr.DataArray) -> xr.DataArray:
        surface_dims = [self.topology.y_dimension, self.topology.x_dimension]
        return utils.linearise_dimensions(data_array, surface_dims)

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xr.Dataset:
        topology = self.topology

        intersecting_indices = [
            item.linear_index
            for polygon, item in self.spatial_index.query(clip_geometry)
            if polygon.intersects(clip_geometry)]
        mask = np.full(topology.shape, fill_value=False)
        mask.ravel()[intersecting_indices] = True

        if buffer > 0:
            mask = masking.blur_mask(mask, size=buffer)
        dimensions = [topology.y_dimension, topology.x_dimension]

        return xr.Dataset(
            data_vars={
                'cell_mask': xr.DataArray(data=mask, dims=dimensions),
            },
            coords={
                topology.latitude_name: topology.latitude.copy(),
                topology.longitude_name: topology.longitude.copy(),
            },
            attrs={'type': 'CFGrid mask'},
        )

    def apply_clip_mask(self, clip_mask: xr.Dataset, work_dir: Pathish) -> xr.Dataset:
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

    def _calculate_bounds(self, values: np.ndarray) -> np.ndarray:
        first_gap = values[1] - values[0]
        last_gap = values[-1] - values[-2]
        mid_points = np.concatenate([
            [values[0] - first_gap / 2],
            (values[1:] + values[:-1]) / 2,
            [values[-1] + last_gap / 2]
        ])
        return np.stack([mid_points[:-1], mid_points[1:]], axis=-1)

    @cached_property
    def latitude_bounds(self) -> xr.DataArray:
        """North/south bounds for each latitude point"""
        with suppress(KeyError):
            return self.dataset.data_vars[self.latitude.attrs['bounds']]
        return xr.DataArray(
            data=self._calculate_bounds(self.latitude.values),
            dims=[self.latitude_name, 'bounds'],
        )

    @cached_property
    def longitude_bounds(self) -> xr.DataArray:
        """East/west bounds for each longitude point"""
        with suppress(KeyError):
            return self.dataset.data_vars[self.longitude.attrs['bounds']]
        return xr.DataArray(
            data=self._calculate_bounds(self.longitude.values),
            dims=[self.longitude_name, 'bounds'],
        )


class CFGrid1D(CFGrid[CFGrid1DTopology]):
    """A :class:`.Format` subclass representing datasets on an axis-aligned grid
    that follows the CF metadata conventions
    and has one dimensional coordinates.
    """
    topology_class = CFGrid1DTopology

    @classmethod
    def check_dataset(cls, dataset: xr.Dataset) -> Optional[int]:
        """
        A dataset is a 1D CF grid if it has one dimensional
        latitude and longitude coordinate variables.

        Many other formats extend the CF conventions,
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
    def polygons(self) -> np.ndarray:
        # Keep these as 2D so that we can easily map centre->grid indices
        longitude_bounds = self.topology.longitude_bounds.values.tolist()
        latitude_bounds = self.topology.latitude_bounds.values.tolist()

        def cell(index: int) -> Polygon:
            y, x = self.unravel_index(index)
            lon_min, lon_max = sorted(longitude_bounds[x])
            lat_min, lat_max = sorted(latitude_bounds[y])
            return box(lon_min, lat_min, lon_max, lat_max)

        # Make a polygon for each wet cell
        return np.array([cell(index) for index in range(self.topology.size)], dtype=object)

    @cached_property
    def face_centres(self) -> np.ndarray:
        topology = self.topology
        xx, yy = np.meshgrid(topology.longitude.values, topology.latitude.values)
        centres = np.column_stack((xx.flatten(), yy.flatten()))
        return cast(np.ndarray, centres)


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


class CFGrid2D(CFGrid[CFGrid2DTopology]):
    """A :class:`.Format` subclass representing datasets on a curvilinear grid
    that follows the CF metadata conventions
    and has two dimensional coordinates.
    """
    topology_class = CFGrid2DTopology

    @classmethod
    def check_dataset(cls, dataset: xr.Dataset) -> Optional[int]:
        """
        A dataset is a 2D CF grid if it has two dimensional
        latitude and longitude coordinate variables.

        Many other formats extend the CF conventions,
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
    def polygons(self) -> np.ndarray:
        xs = self.topology.longitude.values
        ys = self.topology.latitude.values

        # Make a grid around the cell centres
        # by averaging the surrounding centre coordinates.
        # On the edges where there are insufficient 'surrounding' cells,
        # the centres are used.
        # Edge and corner cells will be smaller than the surrounding cells
        # because of this.
        #
        # np.nanmean will return nan for an all-nan column.
        # This is the exact behaviour that we want.
        # numpy emits a warning that can not be silenced when this happens,
        # so that warning is temporarily ignored.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Mean of empty slice", category=RuntimeWarning)

            y_grid = np.nanmean([
                np.pad(ys, pad, constant_values=np.nan)
                for pad in itertools.product([(1, 0), (0, 1)], [(1, 0), (0, 1)])
            ], axis=0)
            x_grid = np.nanmean([
                np.pad(xs, pad, constant_values=np.nan)
                for pad in itertools.product([(1, 0), (0, 1)], [(1, 0), (0, 1)])
            ], axis=0)

        shape = self.topology.shape

        def cell(index: int) -> Optional[Polygon]:
            (j, i) = np.unravel_index(index, shape)
            v1 = x_grid[j, i], y_grid[j, i]
            v2 = x_grid[j, i + 1], y_grid[j, i + 1]
            v3 = x_grid[j + 1, i + 1], y_grid[j + 1, i + 1]
            v4 = x_grid[j + 1, i], y_grid[j + 1, i]
            points = [v1, v2, v3, v4, v1]
            # Can't construct polygons if we don't have all the points
            if np.isnan(points).any():
                return None
            # There is no guarantee that the x or y dimensions are oriented in
            # any particular direction, so the winding order of the polygon is
            # not guaranteed. `orient` will fix this for us so we don't have to
            # think about it.
            return orient(Polygon(points))

        # Make a polygon for each wet cell
        polygons = list(map(cell, range(self.topology.size)))
        return np.array(polygons, dtype=object)

    @cached_property
    def face_centres(self) -> np.ndarray:
        centres = np.column_stack((
            self.make_linear(self.topology.longitude).values,
            self.make_linear(self.topology.latitude).values,
        ))
        return cast(np.ndarray, centres)
