import abc
import dataclasses
import enum
import logging
import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

import numpy
import xarray
from shapely import unary_union
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from emsarray import utils
from emsarray.compat.shapely import SpatialIndex
from emsarray.exceptions import NoSuchCoordinateError
from emsarray.operations import depth, point_extraction
from emsarray.plot import (
    _requires_plot, animate_on_figure, make_plot_title, plot_on_figure,
    polygons_to_collection
)
from emsarray.state import State
from emsarray.types import Bounds, DataArrayOrName, Pathish

if TYPE_CHECKING:
    # Import these optional dependencies only during type checking
    from cartopy.crs import CRS
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver

logger = logging.getLogger(__name__)


#: Some type that can enumerate the different :ref:`grid types <grids>`
#: present in a dataset.
#: This can be an :class:`enum.Enum` listing each different kind of grid.
#:
#: :data:`Index` values will be included in the feature properties
#: of exported geometry from :mod:`emsarray.operations.geometry`.
#: If the index type includes the grid kind,
#: the grid kind needs to be JSON serializable.
#: The easiest way to achieve this is to make your GridKind type subclass :class:`str`:
#:
#: .. code-block:: python
#:
#:     class MyGridKind(str, enum.Enum):
#:         face = 'face'
#:         edge = 'edge'
#:         node = 'node'
#:
#: For cases where the convention only supports a single grid,
#: a singleton enum can be used.
#:
#: More esoteric cases involving datasets with a potentially unbounded numbers of grids
#: can use a type that supports this instead.
GridKind = TypeVar("GridKind")

#: An :ref:`index <indexing>` to a specific point on a grid in this convention.
#: For conventions with :ref:`multiple grids <grids>` (e.g. cells, edges, and nodes),
#: this should be a tuple whos first element is :data:`.GridKind`.
#: For conventions with a single grid, :data:`.GridKind` is not required.
Index = TypeVar("Index")


@dataclasses.dataclass
class SpatialIndexItem(Generic[Index]):
    """Information about an item in the :class:`~shapely.strtree.STRtree`
    spatial index for a dataset.

    See Also
    --------
    Convention.spatial_index
    """

    #: The linear index of this cell
    linear_index: int

    #: The native index of this cell
    index: Index

    #: The geographic shape of this cell
    polygon: Polygon

    def __repr__(self) -> str:
        items = {
            'index': f'{self.index}/{self.linear_index}',
            'polygon': self.polygon.wkt,
        }
        item_str = ' '.join(f'{key}: {value}' for key, value in items.items())
        return f'<{type(self).__name__} {item_str}>'

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SpatialIndexItem):
            return NotImplemented

        # SpatialIndexItems are only for cells / polygons, so we only need to
        # compare the linear indexes. The polygon attribute is not orderable,
        # so comparing on that is going to be unpleasant.
        return self.linear_index < other.linear_index


class Specificity(enum.IntEnum):
    """
    How specific a match is when autodetecting a convention.
    Matches with higher specificity will be prioritised.

    General conventions such as CF Grid are low specificity,
    as many conventions extend and build on CF Grid conventions.

    The SHOC conventions extend the CF grid conventions,
    so a SHOC file will be detected as both CF Grid and SHOC.
    :class:`.ShocStandard` should return a higher specificity
    so that the correct convention implementation is used.
    """
    LOW = 10
    MEDIUM = 20
    HIGH = 30


class Convention(abc.ABC, Generic[GridKind, Index]):
    """
    Each supported geometry convention represents data differently.
    The :class:`Convention` class abstracts these differences away,
    allowing developers to query and manipulate datasets
    without worrying about the details.
    See :ref:`Supported dataset conventions <supported_conventions>`
    for a list of implemented conventions.

    All conventions have the concept of a cell at a geographic location,
    vertically stacked layers of cells,
    and multiple timesteps of data.
    A convention may support additional grids, such as face edges and vertices.
    Refer to :ref:`grids` for more information.

    A cell can be addressed using a linear index or a native index.
    A linear index is always an :class:`int`,
    while the native index type will depend on the specific convention.
    You can convert between a linear and a native index
    using :meth:`.ravel_index` and :meth:`.wind_index`.
    Refer to :ref:`indexing` for more information.
    """
    #: The :class:`xarray.Dataset` instance for this :class:`Convention`
    dataset: xarray.Dataset

    def __init__(self, dataset: xarray.Dataset):
        """
        Make a new convention instance for this dataset.
        """
        self.check_validity(dataset)
        self.dataset = dataset

    @classmethod
    def check_validity(cls, dataset: xarray.Dataset) -> None:
        """Checks that the dataset is OK to use.
        Called during ``__init__``, and raises an exception if the dataset has problems.
        """
        pass  # Subclasses can override this. By default, no checks are made

    @classmethod
    @abc.abstractmethod
    def check_dataset(cls, dataset: xarray.Dataset) -> int | None:
        """
        Check if a dataset uses this convention.

        This may check for variables of the correct dimensions,
        the presence of specific attributes,
        or the layout of the dataset dimensions.

        Specific subclasses must implement this function.
        It will be called by the convention autodetector
        when guessing the correct convention for a dataset.

        If the dataset matches, the return value indicates how specific the match is.
        When autodetecting the correct convention implementation
        the convention with the highest specicifity will be used.
        Many conventions extend the CF grid conventions,
        so the CF Grid convention classes will match many datasets.
        However this match is very generic.
        A more specific implementation such as SHOC may be supported.
        The SHOC convention implementation should return a higher specicifity
        than the CF grid convention.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset instance to inspect.

        Returns
        -------
        int, optional
            If this convention implementation can handle this dataset
            some integer greater than zero is returned.
            The higher the number, the more specific the support.
            If the dataset does not match this convention, None is returned.
            Values on the :class:`~emsarray.conventions.Specificity` enum
            are used by :mod:`emsarray` itself to indicated specificity.
            New convention implementations are free to use these values,
            or use any integer value.

        Example
        -------
        >>> import xarray
        ... from emsarray.conventions.shoc import ShocStandard
        ... from emsarray.conventions.ugrid import UGrid
        ... dataset = xarray.open_dataset("./tests/datasets/shoc_standard.nc")
        >>> ShocStandard.check_dataset(dataset)
        True
        >>> UGrid.check_dataset(dataset)
        False
        """
        pass

    def bind(self) -> None:
        """
        Bind this :class:`.Convention` instance as the default convention
        for the :class:`xarray.Dataset`.
        This convention instance will be assigned to :attr:`dataset.ems`.

        You can use a Convention instance without binding it to a Dataset,
        binding is only necessary if you need to use the :attr:`dataset.ems` accessor.

        .. note::

            If you use :func:`emsarray.open_dataset` or :attr:`dataset.ems`
            to autodetect the dataset convention you do not need to call this method.
            :meth:`Convention.bind` is only useful if you manually construct a :class:`Convention`.

        Examples
        --------

        .. code-block:: python

            # Open a dataset built using the GRASS convention
            dataset = xarray.open_dataset("grass-dataset.nc")

            # Construct a Grass instance for the dataset and bind it
            convention = Grass(dataset)
            convention.bind()

            # dataset.ems is now the bound convention
            assert dataset.ems is convention

        If the dataset already has a bound convention, an error is raised.
        To bind a new convention to a dataset, make a copy of the dataset first:

        .. code-block:: python

            new_dataset = dataset.copy()
            convention = Grass(new_dataset)
            convention.bind()
        """
        state = State.get(self.dataset)
        if state.is_bound():
            raise ValueError(
                "A convention has already been bound to this dataset, "
                "cannot assign a new convention.")
        state.bind_convention(self)

    @utils.deprecated(
        (
            "Convention._get_data_array() has been deprecated. "
            "Use emsarray.utils.name_to_data_array() instead."
        ),
        DeprecationWarning,
    )
    def _get_data_array(self, data_array: DataArrayOrName) -> xarray.DataArray:
        return utils.name_to_data_array(self.dataset, data_array)

    @cached_property
    def time_coordinate(self) -> xarray.DataArray:
        """
        xarray.DataArray: The time coordinate for this dataset.

        Raises
        ------
        exceptions.NoSuchCoordinateError
            If no time coordinate was found

        Notes
        -----
        The CF Conventions state that
        a time variable is defined by having a `units` attribute
        formatted according to the UDUNITS package [1]_.

        xarray will find all time variables and convert them to numpy datetimes.

        References
        ----------
        .. [1] `CF Conventions v1.10, 4.4 Time Coordinate <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#time-coordinate>`_
        """
        for name in self.dataset.variables.keys():
            variable = self.dataset[name]
            # xarray will automatically decode all time variables
            # and move the 'units' attribute over to encoding to store this change.
            if 'units' in variable.encoding:
                units = variable.encoding['units']
                # A time variable must have units of the form '<units> since <epoc>'
                if 'since' in units:
                    # The variable must now be a numpy datetime
                    if variable.dtype.type == numpy.datetime64:
                        return variable
        raise NoSuchCoordinateError("Could not find time coordinate in dataset")

    @cached_property
    def depth_coordinate(self) -> xarray.DataArray:
        """
        xarray.DataArray: The depth coordinate for this dataset.
        For datasets with multiple depth coordinates
        this will be the depth coordinate with the smallest :attr:`~xarray.DataArray.size`.

        Raises
        ------
        exceptions.NoSuchCoordinateError
            If no depth coordinate was found

        Notes
        -----
        The CF Conventions state that
        a depth variable is identifiable by units of pressure; or
        the presence of the ``positive`` attribute with value of ``up`` or ``down``
        [2]_.

        In practice, many datasets do not follow this convention.
        In addition to checking for the ``positive`` attribute,
        all coordinates are checked for a ``standard_name: "depth"``,
        ``coordinate_type: "Z"``, or ``axiz: "Z"``.

        See Also
        --------
        :attr:`depth_coordinates`
        :attr:`get_depth_coordinate_for_data_array`

        References
        ----------
        .. [2] `CF Conventions v1.10, 4.3 Vertical (Height or Depth) Coordinate <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#vertical-coordinate>`_
        """
        depth_coordinates = self.depth_coordinates
        if len(depth_coordinates) == 0:
            raise NoSuchCoordinateError("Dataset has no depth coordinate")
        return min(depth_coordinates, key=lambda c: c.size)

    @cached_property
    def depth_coordinates(self) -> tuple[xarray.DataArray, ...]:
        """
        tuple of xarray.DataArray: All the depth coordinate for this dataset.

        See Also
        --------
        :attr:`depth_coordinate`
        :attr:`get_depth_coordinate_for_data_array`
        """
        depth_coordinates = []
        for name in self.dataset.variables.keys():
            data_array = self.dataset[name]

            if not (
                data_array.attrs.get('positive', '').lower() in {'up', 'down'}
                or data_array.attrs.get('axis') == 'Z'
                or data_array.attrs.get('cartesian_axis') == 'Z'
                or data_array.attrs.get('coordinate_type') == 'Z'
                or data_array.attrs.get('standard_name') == 'depth'
            ):
                continue

            try:
                # If the variable is defined on a grid,
                # it is more likely to be a bathymetry variable
                # not the coordinate for the depth layers.
                self.get_grid_kind(data_array)
                continue
            except ValueError:
                # The variable isn't on a grid - this is good!
                pass

            depth_coordinates.append(data_array)

        return tuple(depth_coordinates)

    @utils.deprecated(
        (
            "Convention.get_time_name() is deprecated. "
            "Use Convention.time_coordinate.name instead."
        ),
        DeprecationWarning,
    )
    def get_time_name(self) -> Hashable:
        """Get the name of the time variable in this dataset.

        .. deprecated:: 0.7.0
            :meth:`Convention.get_time_name()` is deprecated and will be removed.
            Use `Convention.time_coordinate.name` instead.

        Returns
        -------
        Hashable
            The name of the time coordinate.

        See Also
        --------
        :attr:`Convention.time_coordinate`

        Raises
        ------
        exceptions.NoSuchCoordinateError
            If no time coordinate was found
        """
        return self.time_coordinate.name

    @utils.deprecated(
        (
            "Convention.get_depth_name() is deprecated. "
            "Use Convention.depth_coordinate.name instead."
        ),
        DeprecationWarning,
    )
    def get_depth_name(self) -> Hashable:
        """Get the name of the layer depth coordinate variable.

        .. deprecated:: 0.7.0
            :meth:`Convention.get_depth_name()` is deprecated and will be removed.
            Use `Convention.depth_coordinate.name` instead.

        For datasets with multiple depth variables, this should be the one that
        represents the centre of the layer, not the bounds.

        Note that this is the name of the coordinate variable,
        not the name of the dimension, for datasets where these differ.

        Returns
        -------
        Hashable
            The name of the depth coordinate.

        Raises
        ------
        exceptions.NoSuchCoordinateError
            If no time coordinate was found

        See Also
        --------
        :attr:`Convention.depth_coordinate`
        :meth:`Convention.get_depth_coordinate_for_data_array`
        """
        return self.depth_coordinate.name

    @utils.deprecated(
        (
            "Convention.get_all_depth_names() is deprecated. "
            "Use [c.name for c in Convention.depth_coordinates] instead."
        ),
        DeprecationWarning,
    )
    def get_all_depth_names(self) -> list[Hashable]:
        """Get the names of all depth layers.
        Some datasets include both a depth layer centre,
        and the depth layer 'edges'.

        .. deprecated:: 0.7.0
            :meth:`Convention.get_all_depth_names()` is deprecated and will be removed.
            Use `[c.name for c in Convention.depth_coordinates]` instead.

        Note that this is the names of the coordinate variables,
        not the names of the dimensions, for datasets where these differ.

        See also
        --------
        :attr:`depth_coordinate`
        :attr:`depth_coordinates`
        :meth:`get_depth_coordinate_for_data_array`
        """
        return [c.name for c in self.depth_coordinates]

    def get_depth_coordinate_for_data_array(
        self,
        data_array: DataArrayOrName,
    ) -> xarray.DataArray:
        """
        Find the depth coordinate for a particular data array.
        Some conventions will contain multiple depth coordinates,
        meaning that a default :attr:`depth_coordinate` value can be misleading.

        Parameters
        ----------
        data_array : xarray.DataArray or Hashable
            A data array or the name of a data array in the dataset.

        Returns
        -------
        xarray.DataArray
            The depth coordinate variable for the data array.

        Raises
        ------
        NoSuchCoordinateError
            If data array does not have an associated depth coordinate
        ValueError
            If multiple depth coordinates matched the data array.

        See also
        --------
        :attr:`depth_coordinate`
            The default or main depth coordinate for this dataset,
            but not necessarily the correct depth coordinate for all variables.
        :attr:`depth_coordinates`
            All the depth coordinates in this dataset.
        """
        data_array = utils.name_to_data_array(self.dataset, data_array)
        name = repr(data_array.name) if data_array.name is not None else 'data array'

        candidates = [
            coordinate
            for coordinate in self.depth_coordinates
            if set(coordinate.dims) <= set(data_array.dims)
        ]
        if len(candidates) == 0:
            raise NoSuchCoordinateError(f"No depth coordinate found for {name}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple possible depth coordinates found for {name}: "
                ", ".join(repr(c.name) for c in candidates)
            )
        return candidates[0]

    @abc.abstractmethod
    def ravel_index(self, index: Index) -> int:
        """Convert a convention native index to a linear index.

        Each conventnion has a different native index type,
        read the specific conventnion documentation for more information.

        Parameters
        ----------
        index : :data:`.Index`
            The native index to ravel.

        Returns
        -------
        :class:`int`
            The linear index for that native index.

        Example
        -------

        If the dataset used the CF Grid conventions,
        across a (latitude, longitude) grid of size (30, 40):

        .. code-block:: python

            >>> temp = dataset.data_vars['temp']
            >>> temp.dims
            ('t', 'z', 'y', 'x')
            >>> temp.shape
            (10, 20, 30, 40)
            >>> dataset.ems.ravel_index((3, 4))
            124

        Cell polygons are indexed in the same order as the linear indexes for cells.
        To find the polygon for the cell with the native index ``(3, 4)``:

        .. code-block:: python

            >>> index = (3, 4)
            >>> linear_index = dataset.ems.ravel_index(index)
            >>> polygon = dataset.ems.polygons[linear_index]

        See Also
        --------
        :meth:`.Convention.wind_index` : The inverse operation
        """
        pass

    @abc.abstractmethod
    def wind_index(
        self,
        linear_index: int,
        *,
        grid_kind: GridKind | None = None,
    ) -> Index:
        """Convert a linear index to a conventnion native index.

        Each conventnion has a different native index type,
        read the specific conventnion documentation for more information.

        Parameters
        ----------
        linear_index : int
            The linear index to wind.
        grid_kind : :data:`.GridKind`, optional
            Used to indicate what kind of index is being wound,
            for conventions with multiple grids.
            Optional, if not provided the default grid kind will be used.

        Returns
        -------
        :data:`.Index`
            The convention native index for that same cell

        Example
        -------

        If the dataset used the CF Grid conventions,
        across a (latitude, longitude) grid of size (30, 40):

        .. code-block:: python

            >>> temp = dataset.data_vars['temp']
            >>> temp.dims
            ('t', 'z', 'y', 'x')
            >>> temp.shape
            (10, 20, 30, 40)
            >>> dataset.ems.wind_index(124)
            (3, 4)

        See Also
        --------
        :meth:`.Convention.ravel_index` : The inverse operation
        """
        pass

    @utils.deprecated(
        (
            "Convention.unravel_index() has been renamed to "
            "Convention.wind_index()."
        ),
        DeprecationWarning,
    )
    def unravel_index(
        self,
        linear_index: int,
        grid_kind: GridKind | None = None,
    ) -> Index:
        """An alias for :meth:`Convention.wind_index()`.

        .. deprecated:: 0.6.0
            Use :meth:`Convention.wind_index()` instead
        """
        return self.wind_index(linear_index, grid_kind=grid_kind)

    @property
    @abc.abstractmethod
    def grid_kinds(self) -> frozenset[GridKind]:
        """
        All of the :data:`grid kinds <.GridKind>` this dataset includes.
        """
        pass

    @property
    @abc.abstractmethod
    def default_grid_kind(self) -> GridKind:
        """
        The default :data:`grid kind <.GridKind>` for this dataset.
        For most datasets this should be the face grid.
        """
        pass

    @property
    @abc.abstractmethod
    def grid_size(self) -> dict[GridKind, int]:
        """The linear size of each grid kind."""
        pass

    @abc.abstractmethod
    def get_grid_kind(self, data_array: xarray.DataArray) -> GridKind:
        """
        Determines the relevant grid kind for this data array.

        If the data array is not indexable using the native index types
        a ValueError is raised.

        Parameters
        ----------
        data_array : xarray.DataArray
            The data array to inspect

        Returns
        -------
        :data:`.GridKind`

        Raises
        ------
        ValueError
            If the data array passed in is not indexable using any native index type
            a ValueError is raised.
            Depth coordinates or time coordinates are examples of data arrays
            that will not be indexable and will raise an error.

        Example
        -------
        For a :class:`UGRID <.ugrid.UGrid>` dataset
        with temperature data defined at the cell centres
        and current defined as flux through the cell edges:

        .. code-block:: python

            >>> dataset.data_vars['temp'].dims
            ('time', 'depth', 'face')
            >>> dataset.data_vars['u1'].dims
            ('time', 'depth', 'edge')
            >>> dataset.ems.get_grid_kind(dataset.data_vars['temp'])
            UGridKind.face
            >>> dataset.ems.get_grid_kind(dataset.data_vars['u1'])
            UGridKind.edge
        """
        pass

    def get_grid_kind_and_size(
        self, data_array: xarray.DataArray,
    ) -> tuple[GridKind, int]:
        """
        Determines the relevant index kind and the extent of the linear index space
        for this data array.

        .. deprecated:: 0.6.0
            This method is replaced by :meth:`Convention.get_grid_kind()`
            and :attr:`Convention.grid_size`.

        If the data array is not indexable using the native index types
        a ValueError is raised.

        Parameters
        ----------
        data_array : xarray.DataArray
            The data array to introspect

        Returns
        -------
        tuple of :data:`.GridKind` and int

        Raises
        ------
        ValueError
            If the data array passed in is not indexable using any native index type
            a ValueError is raised.
            Depth coordinates or time coordinates are examples of data arrays
            that will not be indexable and will raise an error.

        Example
        -------
        For a :class:`UGRID <.ugrid.UGrid>` dataset
        with temperature data defined at the cell centres
        and current defined as flux through the cell edges:

        .. code-block:: python

            >>> dataset.ems.topology.face_count
            4
            >>> dataset.ems.topology.edge_count
            9
            >>> dataset.data_vars['temp'].dims
            ('time', 'depth', 'face')
            >>> dataset.data_vars['u1'].dims
            ('time', 'depth', 'edge')
            >>> dataset.ems.get_grid_kind_and_size(dataset.data_vars['temp'])
            (UGridKind.face, 4)
            >>> dataset.ems.get_grid_kind_and_size(dataset.data_vars['u1'])
            (UGridKind.edge, 9)
        """
        grid_kind = self.get_grid_kind(data_array)
        size = self.grid_size[grid_kind]
        return grid_kind, size

    @abc.abstractmethod
    def ravel(
        self,
        data_array: xarray.DataArray,
        *,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        """
        Flatten the surface dimensions of a :class:`~xarray.DataArray`,
        returning a flatter :class:`numpy.ndarray` indexed in the same order as the linear index.

        For DataArrays with extra dimensions such as time or depth,
        only the surface dimensions are flattened.
        Other dimensions are left as is.

        For datasets with multiple grids,
        with data defined on edges or vertices for example,
        this will flatten those data arrays in the correct linear order
        to be indexed by the relevant index type.

        Parameters
        ----------
        data_array : xarray.DataArray
            One of the data variables from this dataset.
        linear_dimension : Hashable, optional
            The name of the new dimension to flatten the surface dimensions to.
            Defaults to 'index'.

        Returns
        -------
        xarray.DataArray
            A new data array, where all the surface dimensions
            have been flattened in to one linear array.
            The values for each cell, in the same order as the linear index for this dataset.
            Any other dimensions, such as depth or time, will be retained.

        See Also
        --------
        .utils.ravel_dimensions : A function that ravels some given dimensions in a dataset.
        Convention.wind : The inverse operation.
        """
        pass

    @abc.abstractmethod
    def wind(
        self,
        data_array: xarray.DataArray,
        *,
        grid_kind: GridKind | None = None,
        axis: int | None = None,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        """
        Wind a flattened :class:`~xarray.DataArray`
        so that it has the same shape as data variables in this dataset.

        By using :attr:`.grid_size` and :meth:`.wind()` together
        it is possible to construct new data variables for a dataset
        of any arbitrary shape.

        Parameters
        ----------
        data_array : xarray.DataArray
            One of the data variables from this dataset.
        grid_kind : GridKind
            The kind of grid this data array represents,
            for those conventions with multiple grid kinds.
            Optional, defaults to the default grid kind.
        axis : int, optional
            The axis number that should be wound.
            Optional, defaults to the last axis.
        linear_dimension : Hashable, optional
            The axis number that should be wound.
            Optional, defaults to the last dimension.

        Returns
        -------
        xarray.DataArray
            A new data array where the linear data have been wound
            to match the shape of the convention.
            Any other dimensions, such as depth or time, will be retained.

        Examples
        --------
        The following will construct a data array of the correct shape
        for any convention supported by emsarray:

        .. code-block:: python

            import emsarray
            import numpy
            import xarray

            dataset = emsarray.tutorial.open_dataset('kgari')
            face_size = dataset.ems.grid_size[dataset.ems.default_grid_kind]
            flat_array = xarray.DataArray(
                data=numpy.arange(face_size),
                dims=['index'],
            )
            data_array = dataset.ems.wind(flat_array)

        This will construct a boolean array indicating
        which cells of a dataset intersect a target geometry:

        .. code-block:: python

            import emsarray
            import numpy
            import shapely
            import xarray

            dataset = emsarray.tutorial.open_dataset('gbr4')
            target = shapely.Polygon([
                [152.8088379, -22.7863108],
                [153.9184570, -22.2280904],
                [153.4680176, -20.9614396],
                [151.8255615, -20.4012720],
                [151.4135742, -21.8309067],
                [152.0068359, -22.4313402],
                [152.8088379, -22.7863108],
            ])

            hits = dataset.ems.strtree.query(target, predicate='intersects')
            grid_size = dataset.ems.grid_size[dataset.ems.default_grid_kind]
            intersecting_cells = xarray.DataArray(
                data=numpy.zeros(grid_size, dtype=bool),
                dims=['index'],
            )
            intersecting_cells.values[hits] = True
            intersecting_cells = dataset.ems.wind(intersecting_cells)

        See Also
        --------
        .utils.wind_dimension : Reshape a particular dimension in a data array.
        Convention.ravel : The inverse operation.
        """
        pass

    @utils.deprecated(
        "Convention.make_linear() has been renamed to Convention.ravel().",
        DeprecationWarning,
    )
    def make_linear(self, data_array: xarray.DataArray) -> xarray.DataArray:
        """A deprecated alias for :meth:`Convention.ravel()`

        .. deprecated:: 0.6.0
            This method is replaced by
            :meth:`Convention.ravel()`
        """
        return self.ravel(data_array)

    @cached_property  # type: ignore
    @_requires_plot
    def data_crs(self) -> 'CRS':
        """
        The coordinate reference system that coordinates in this dataset are
        defined in.
        Used by :meth:`.Convention.make_poly_collection` and :meth:`.Convention.make_quiver`.
        Defaults to :class:`cartopy.crs.PlateCarree`.
        """
        # Lazily imported here as cartopy is an optional dependency
        from cartopy.crs import PlateCarree
        return PlateCarree()

    @_requires_plot
    def plot_on_figure(
        self,
        figure: 'Figure',
        scalar: DataArrayOrName | None = None,
        vector: tuple[DataArrayOrName, DataArrayOrName] | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot values for a :class:`~xarray.DataArray`
        on a :mod:`matplotlib` :class:`~matplotlib.figure.Figure`.

        The data array can either be passed in directly,
        or the name of a data array on this :attr:`Convention.dataset` instance.
        The data array does not have to come from the same dataset,
        as long as the dimensions are the same.

        This method will only plot a single time step and depth layer.
        Callers are responsible for selecting a single slice before calling this method.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The :class:`~matplotlib.figure.Figure` instance to plot this on.
        scalar : DataArrayOrName
            The :class:`~xarray.DataArray` to plot,
            or the name of an existing DataArray in this Dataset.
        vector : tuple of DataArrayOrName
            A tuple of the *u* and *v* components of a vector.
            The components should be a :class:`~xarray.DataArray`,
            or the name of an existing DataArray in this Dataset.
        **kwargs
            Any extra keyword arguments are passed on to
            :meth:`emsarray.plot.plot_on_figure`

        See Also
        --------
        :func:`.plot.plot_on_figure` : The underlying implementation
        """
        if scalar is not None:
            kwargs['scalar'] = utils.name_to_data_array(self.dataset, scalar)

        if vector is not None:
            kwargs['vector'] = tuple(utils.name_to_data_array(self.dataset, v) for v in vector)

        if title is not None:
            kwargs['title'] = title
        elif scalar is not None and vector is None:
            # Make a title out of the scalar variable, but only if a title
            # hasn't been supplied and we don't also have vectors to plot.
            #
            # We can't make a good name from vectors,
            # as they are in two variables with names like
            # 'u component of current' and 'v component of current'.
            #
            # Users can supply their own titles
            # if this automatic behaviour is insufficient
            kwargs['title'] = make_plot_title(self.dataset, kwargs['scalar'])

        plot_on_figure(figure, self, **kwargs)

    @_requires_plot
    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Plot a data array and automatically display it.

        This method is most useful when working in Jupyter notebooks
        which display figures automatically.
        This method is a wrapper around :meth:`.plot_on_figure`
        that creates and shows a :class:`~matplotlib.figure.Figure` for you.
        All arguments are passed on to :meth:`.plot_on_figure`,
        refer to that function for details.

        See Also
        --------
        :meth:`.plot_on_figure`
        """
        from matplotlib import pyplot
        self.plot_on_figure(pyplot.figure(), *args, **kwargs)
        pyplot.show()

    @_requires_plot
    def animate_on_figure(
        self,
        figure: 'Figure',
        scalar: DataArrayOrName | None = None,
        vector: tuple[DataArrayOrName, DataArrayOrName] | None = None,
        coordinate: DataArrayOrName | None = None,
        title: str | Callable[[Any], str] | None = None,
        **kwargs: Any,
    ) -> 'FuncAnimation':
        """
        Make an animated plot of a data array.

        For real world examples, refer to the ``examples/animation.ipynb`` notebook.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The :class:`matplotlib.figure.Figure` to plot the animation on
        data_array : DataArrayOrName
            The :class:`xarray.DataArray` to plot.
            If a string is passed in,
            the variable with that name is taken from :attr:`dataset`.
        coordinate : Hashable or xarray.DataArray, optional
            The coordinate to vary across the animation.
            Pass in either the name of a coordinate variable
            or coordinate variable itself.
            Optional, if not supplied the time coordinate
            from :attr:`Convention.time_coordinate` is used.
            Other appropriate coordinates to animate over include depth.
        **kwargs
            Any extra arguments are passed to :func:`.plot.animate_on_figure`.

        Returns
        -------
        :class:`matplotlib.animation.Animation`
            The data variable plotted as an animation on the figure.
            Call :meth:`Animation.save() <matplotlib.animation.Animation.save>`
            to save animation to a file, or display it in a Notebook using
            :meth:`Animation.to_jshtml() <matplotlib.animation.Animation.to_jshtml>`.

        See Also
        --------
        :func:`.plot.animate_on_figure`
        """

        if coordinate is None:
            # Assume the user wants to plot along the time axis by default.
            coordinate = self.time_coordinate
        else:
            coordinate = utils.name_to_data_array(self.dataset, coordinate)

        if len(coordinate.dims) != 1:
            raise ValueError("Coordinate variable must be one dimensional")

        coordinate_dim = coordinate.dims[0]

        if scalar is not None:
            scalar = utils.name_to_data_array(self.dataset, scalar)
            if coordinate_dim not in scalar.dims:
                raise ValueError("Scalar dimensions do not match coordinate axis to animate along")
            kwargs['scalar'] = scalar

        if vector is not None:
            vector = (
                utils.name_to_data_array(self.dataset, vector[0]),
                utils.name_to_data_array(self.dataset, vector[1]),
            )
            if not all(coordinate_dim in component.dims for component in vector):
                raise ValueError("Vector dimensions do not match coordinate axis to animate along")
            kwargs['vector'] = vector

        if title is not None:
            kwargs['title'] = title
        elif scalar is not None and vector is None:
            # Make a title out of the scalar variable, but only if a title
            # hasn't been supplied and we don't also have vectors to plot.
            #
            # We can't make a good name from vectors,
            # as they are in two variables with names like
            # 'u component of current' and 'v component of current'.
            #
            # Users can supply their own titles
            # if this automatic behaviour is insufficient
            title_bits = []
            if 'long_name' in scalar.attrs:
                title_bits.append(str(scalar.attrs['long_name']))
            elif scalar.name:
                title_bits.append(str(scalar.name))

            if 'long_name' in coordinate.attrs:
                title_bits.append(str(coordinate.attrs['long_name']) + ': {}')
            elif coordinate.name:
                title_bits.append(str(coordinate.name) + ': {}')
            else:
                title_bits.append('{}')
            kwargs['title'] = '\n'.join(title_bits)

        return animate_on_figure(figure, self, coordinate=coordinate, **kwargs)

    @_requires_plot
    @utils.timed_func
    def make_poly_collection(
        self,
        data_array: DataArrayOrName | None = None,
        **kwargs: Any,
    ) -> 'PolyCollection':
        """
        Make a :class:`~matplotlib.collections.PolyCollection`
        from the geometry of this :class:`~xarray.Dataset`.
        This can be used to make custom matplotlib plots from your data.

        If a :class:`~xarray.DataArray` is passed in,
        the values of that are assigned to the PolyCollection `array` parameter.

        Parameters
        ----------
        data_array : Hashable or :class:`xarray.DataArray`, optional
            A data array, or the name of a data variable in this dataset. Optional.
            If given, the data array is :meth:`ravelled <.ravel>`
            and passed to :meth:`PolyCollection.set_array() <matplotlib.cm.ScalarMappable.set_array>`.
            The data is used to colour the patches.
            Refer to the matplotlib documentation for more information on styling.
        **kwargs
            Any keyword arguments are passed to the
            :class:`~matplotlib.collections.PolyCollection` constructor.

        Returns
        -------
        :class:`~matplotlib.collections.PolyCollection`
            A PolyCollection constructed using the geometry of this dataset.

        Example
        -------

        .. code-block:: python

            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt
            import emsarray

            figure = plt.figure(figsize=(10, 8))
            axes = plt.subplot(projection=ccrs.PlateCarree())
            axes.set_aspect(aspect='equal', adjustable='datalim')

            ds = emsarray.open_dataset("./tests/datasets/ugrid_mesh2d.nc")
            ds = ds.isel(record=0, Mesh2_layers=-1)
            patches = ds.ems.make_poly_collection('temp')
            axes.add_collection(patches)
            figure.colorbar(patches, ax=axes, location='right', label='meters')

            axes.set_title("Depth")
            axes.autoscale()
            figure.show()
        """
        if data_array is not None:
            if 'array' in kwargs:
                raise TypeError(
                    "Can not pass both `data_array` and `array` to make_poly_collection"
                )

            data_array = utils.name_to_data_array(self.dataset, data_array)

            data_array = self.ravel(data_array)
            if len(data_array.dims) > 1:
                raise ValueError(
                    "Data array has too many dimensions - did you forget to "
                    "select a single timestep or a single depth layer?")

            values = data_array.values[self.mask]
            kwargs['array'] = values
            if 'clim' not in kwargs:
                kwargs['clim'] = (numpy.nanmin(values), numpy.nanmax(values))

        if 'transform' not in kwargs:
            kwargs['transform'] = self.data_crs

        return polygons_to_collection(self.polygons[self.mask], **kwargs)

    def make_patch_collection(
        self,
        data_array: DataArrayOrName | None = None,
        **kwargs: Any,
    ) -> 'PolyCollection':
        warnings.warn(
            "Convention.make_patch_collection has been renamed to "
            "Convention.make_poly_collection, and now returns a PolyCollection",
            category=DeprecationWarning,
        )
        return self.make_poly_collection(data_array, **kwargs)

    @_requires_plot
    def make_quiver(
        self,
        axes: 'Axes',
        u: DataArrayOrName | None = None,
        v: DataArrayOrName | None = None,
        **kwargs: Any,
    ) -> 'Quiver':
        """
        Make a :class:`matplotlib.quiver.Quiver` instance to plot vector data.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes to make this quiver on.
        u, v : xarray.DataArray or str, optional
            The DataArrays or the names of DataArrays in this dataset
            that make up the *u* and *v* components of the vector.
            If omitted, a Quiver will be constructed with all components set to 0.
        **kwargs
            Any keyword arguments are passed on to the Quiver constructor.

        Returns
        -------
        matplotlib.quiver.Quiver
            A quiver instance that can be added to a plot
        """
        from matplotlib.quiver import Quiver

        x, y = numpy.transpose(self.face_centres)

        # A Quiver needs some values when being initialized.
        # We don't always want to provide values to the quiver,
        # sometimes preferring to fill them in later,
        # so `u` and `v` are optional.
        # If they are not provided, we set default quiver values of `numpy.nan`.
        values: tuple[numpy.ndarray, numpy.ndarray] | tuple[float, float]
        values = numpy.nan, numpy.nan

        if u is not None and v is not None:
            u = utils.name_to_data_array(self.dataset, u)
            v = utils.name_to_data_array(self.dataset, v)

            if u.dims != v.dims:
                raise ValueError(
                    "Vector data array dimensions must be identical!\n"
                    f"u dimensions: {tuple(u.dims)}\n"
                    f"v dimensions: {tuple(v.dims)}"
                )

            u, v = self.ravel(u), self.ravel(v)

            if len(u.dims) > 1:
                raise ValueError(
                    "Vector data arrays have too many dimensions - did you forget to "
                    "select a single timestep or a single depth layer?")

            values = u.values, v.values

        if 'transform' not in kwargs:
            kwargs['transform'] = self.data_crs

        return Quiver(axes, x, y, *values, **kwargs)

    @property
    @abc.abstractmethod
    def polygons(self) -> numpy.ndarray:
        """A :class:`numpy.ndarray` of :class:`shapely.Polygon` instances
        representing the cells in this dataset.

        The order of the polygons in the list
        corresponds to the linear index of this dataset.
        Not all valid cell indexes have a polygon,
        these holes are represented as :data:`None` in the list.
        If you want a list of just polygons, apply the :attr:`mask <Convention.mask>`:

        .. code-block:: python

            dataset = emsarray.open_dataset("...")
            only_polygons = dataset.ems.polygons[dataset.ems.mask]

        See Also
        --------
        :meth:`ravel_index`
        :attr:`mask`
        """
        pass

    @cached_property
    def face_centres(self) -> numpy.ndarray:
        """
        A numpy :class:`~numpy.ndarray` of face centres, which are (x, y) pairs.
        The first dimension will be the same length and in the same order
        as :attr:`Convention.polygons`,
        while the second dimension will always be of size 2.
        """
        # This default implementation simply finds the centroid of each polygon.
        # Subclasses are free to override this if the particular convention and dataset
        # provides the cell centres as a data array.
        centres = numpy.array([
            polygon.centroid.coords[0] if polygon is not None else [numpy.nan, numpy.nan]
            for polygon in self.polygons
        ])
        return cast(numpy.ndarray, centres)

    @cached_property
    def mask(self) -> numpy.ndarray:
        """
        A boolean :class:`numpy.ndarray` indicating which cells have valid polygons.
        This can be used to select only items from linear arrays
        that have a corresponding polygon.

        .. code-block:: python

            dataset = emsarray.open_dataset("...")
            mask = dataset.ems.mask
            plottable_polygons = dataset.ems.polygons[mask]
            plottable_values = dataset.ems.ravel("eta")[mask]

        See Also
        --------
        :meth:`Convention.ravel`
        """
        mask = numpy.fromiter(
            (p is not None for p in self.polygons),
            dtype=bool, count=self.polygons.size)
        return cast(numpy.ndarray, mask)

    @cached_property
    def geometry(self) -> Polygon | MultiPolygon:
        """
        A :class:`shapely.Polygon` or :class:`shapely.MultiPolygon` that represents
        the geometry of the entire dataset.

        This is equivalent to the union of all polygons in the dataset,
        although specific conventions may have a simpler way of constructing this.
        """
        return unary_union(self.polygons[self.mask])

    @cached_property
    def bounds(self) -> Bounds:
        """
        Returns minimum bounding region (minx, miny, maxx, maxy) of the entire dataset.

        This is equivalent to the bounds of the dataset :attr:`geometry`,
        although specific conventons may have a simpler way of constructing this.
        """
        return cast(Bounds, self.geometry.bounds)

    @cached_property
    @utils.timed_func
    def strtree(self) -> STRtree:
        """
        A :class:`shapely.strtree.STRtree` spatial index of all cells in this dataset.
        This allows for fast spatial lookups, querying which cells lie at
        a point, or which cells intersect a geometry.

        Querying the STRtree will return the linear indexes of any matching cells.
        Use :attr:`polygons` to find the geometries associated with each index.
        Use :meth:`wind_index()` to transform this back to a native index,
        or :meth:`ravel` to linearise a variable.

        Examples
        --------

        Find the indexes of all cells that intersect a line:

        .. code-block:: python

            dataset = emsarray.tutorial.open_dataset('austen')
            geometry = shapely.linestring([
                [146.9311523, -15.7076628],
                [149.3261719, -19.0413488],
                [152.2485352, -21.6778483],
                [154.2480469, -24.2469646],
                [155.1049805, -27.5082714],
                [154.7753906, -31.2973280],
                [153.3911133, -34.1072564],
                [152.0947266, -36.0846213],
                [150.6005859, -38.7712164],
            ])
            hits = dataset.ems.strtree.query(geometry, predicate='intersects')
        """
        return STRtree(self.polygons)

    @cached_property
    @utils.timed_func
    @utils.deprecated(
        (
            "Convention.spatial_index is deprecated. "
            "Use Convention.strtree instead."
        ),
        DeprecationWarning,
    )
    def spatial_index(self) -> SpatialIndex[SpatialIndexItem[Index]]:
        """
        A :class:`shapely.strtree.STRtree` spatial index of all cells in this dataset.
        This allows for fast spatial lookups, querying which cells lie at
        a point, or which cells intersect a geometry.

        .. deprecated:: 0.6.0

            Use :attr:`Convention.strtree` instead.

            This existed as a wrapper around a Shapely STRtree,
            which changed its interface in Shapely 2.0.
            Shapely 1.8.x is no longer supported by emsarray
            so this compatibility wrapper is deprecated.
            Use :attr:`Convention.strtree` directly instead.

        Querying this spatial index will return a list of
        (:class:`polygon <shapely.geometry.Polygon>`, :class:`SpatialIndexItem`) tuples
        corresponding to each matching cell.
        SpatialIndexItem instances have the cells linear index, native index, and polygon.

        The query results from the STRtree contain all geometries with overlapping bounding boxes.
        Query results need to be refined further
        by comparing the cell geometry to the query geometry.
        Refer to the Shapely 1.8.x STRtree docs for examples.

        See Also
        --------
        :class:`.SpatialIndexItem` : The dataclass returned from querying the STRtree.

        `Shapely 1.8.x STRtree docs <https://shapely.readthedocs.io/en/1.8.5.post1/manual.html#str-packed-r-tree>`_
        """
        items = [
            (poly, SpatialIndexItem(index, self.wind_index(index), poly))
            for index, poly in enumerate(self.polygons)
            if poly is not None
        ]
        return SpatialIndex(items)

    def get_index_for_point(
        self,
        point: Point,
    ) -> SpatialIndexItem[Index] | None:
        """
        Find the index for a :class:`~shapely.Point` in the dataset.

        Parameters
        ----------
        point : shapely.Point
            The geographic point to query

        Returns
        -------
        :class:`SpatialIndexItem`, optional
            The :class:`SpatialIndexItem` for the point queried.
            This indicates the polygon that intersected the point
            and the index of that polygon in the dataset.

            If the point does not intersect the dataset, None is returned.

        Notes
        -----
        In the case where the point intersects multiple cells
        the cell with the lowest linear index is returned.
        This can happen if the point is exactly one of the cell vertices,
        or falls on a cell edge,
        or if the geometry of the dataset contains overlapping polygons.
        """
        hits = numpy.sort(self.strtree.query(point, predicate='intersects'))
        if len(hits) > 0:
            linear_index = hits[0]
            return SpatialIndexItem(
                linear_index=linear_index,
                index=self.wind_index(linear_index),
                polygon=self.polygons[linear_index])
        return None

    def selector_for_index(self, index: Index) -> xarray.Dataset:
        """
        Convert a convention native index into a selector
        that can be passed to :meth:`Dataset.isel <xarray.Dataset.isel>`.

        Parameters
        ----------
        index : :data:`Index`
            A convention native index

        Returns
        -------
        selector : xarray.Dataset
            A selector suitable for passing to :meth:`xarray.Dataset.isel`
            that will select values at this index.

        See Also
        --------
        :meth:`.select_index`
        :meth:`.select_point`
        :meth:`.selector_for_indexes`
        :ref:`indexing`

        Notes
        -----

        The returned selector is an :class:`xarray.Dataset`,
        but the contents of the dataset are dependent on the specific convention
        and may change between versions of emsarray.
        The selector should be treated as an opaque value
        that is only useful when passed to :meth:`xarray.Dataset.isel()`.
        """
        index_dimension = utils.find_unused_dimension(self.dataset, 'index')
        dataset = self.selector_for_indexes([index], index_dimension=index_dimension)
        dataset = dataset.squeeze(dim=index_dimension, drop=False)
        return dataset

    @abc.abstractmethod
    def selector_for_indexes(
        self,
        indexes: list[Index],
        *,
        index_dimension: Hashable | None = None,
    ) -> xarray.Dataset:
        """
        Convert a list of convention native indexes into a selector
        that can be passed to :meth:`Dataset.isel <xarray.Dataset.isel>`.

        Parameters
        ----------
        indexes : list of :data:`Index`
            A list of convention native indexes

        Returns
        -------
        selector : xarray.Dataset
            A selector suitable for passing to :meth:`xarray.Dataset.isel`
            that will select values at this index.

        See Also
        --------
        :meth:`.select_indexes`
        :meth:`.select_points`
        :meth:`.selector_for_index`
        :ref:`indexing`

        Notes
        -----

        The returned selector is an :class:`xarray.Dataset`,
        but the contents of the dataset are dependent on the specific convention
        and may change between versions of emsarray.
        The selector should be treated as an opaque value
        that is only useful when passed to :meth:`xarray.Dataset.isel()`.
        """
        pass

    def select_index(
        self,
        index: Index,
        drop_geometry: bool = True,
    ) -> xarray.Dataset:
        """
        Return a new dataset that contains values only from a single index.
        This is much like doing a :func:`xarray.Dataset.isel()` on an index,
        but works with convention native index types.

        An index is associated with a grid kind.
        The returned dataset will only contain variables that were defined on this grid,
        with the single indexed point selected.
        For example, if the index of a face is passed in,
        the returned dataset will not contain any variables defined on an edge.

        Parameters
        ----------
        index : :data:`Index`
            The index to select.
        drop_geometry : bool, default True
            Whether to drop geometry variables from the returned point dataset.
            If the geometry data is kept
            the associated geometry data will no longer conform to the dataset convention
            and may not conform to any sensible convention at all.
            The format of the geometry data left after selecting points is convention-dependent.

        Returns
        -------
        :class:`xarray.Dataset`
            A new dataset that is subset to the one index.

        See also
        --------
        :meth:`.select_point`
        :meth:`.select_indexes`

        Notes
        -----

        The returned dataset will most likely not have sufficient coordinate data
        to be used with a particular :class:`Convention` any more.
        The ``dataset.ems`` accessor will raise an error if accessed on the new dataset.
        """
        index_dimension = utils.find_unused_dimension(self.dataset, 'index')
        dataset = self.select_indexes([index], index_dimension=index_dimension, drop_geometry=drop_geometry)
        dataset = dataset.squeeze(dim=index_dimension, drop=False)
        return dataset

    def select_indexes(
        self,
        indexes: list[Index],
        *,
        index_dimension: Hashable | None = None,
        drop_geometry: bool = True,
    ) -> xarray.Dataset:
        """
        Return a new dataset that contains values only at the selected indexes.
        This is much like doing a :func:`xarray.Dataset.isel()` on some indexes,
        but works with convention native index types.

        An index is associated with a grid kind.
        The returned dataset will only contain variables that were defined on this grid,
        with the indexed points selected.
        For example, if the index of a face is passed in,
        the returned dataset will not contain any variables defined on an edge.

        Parameters
        ----------
        indexes : list of :data:`Index`
            The indexes to select.
            The indexes must all be for the same grid kind.
        index_dimension : str, optional
            The name of the new dimension added for each index to select.
            Defaults to the :func:`first unused dimension <.utils.find_unused_dimension>` with prefix `index`.
        drop_geometry : bool, default True
            Whether to drop geometry variables from the returned point dataset.
            If the geometry data is kept
            the associated geometry data will no longer conform to the dataset convention
            and may not conform to any sensible convention at all.
            The format of the geometry data left after selecting points is convention-dependent.

        Returns
        -------
        :class:`xarray.Dataset`
            A new dataset that is subset to the indexes.

        See also
        --------
        :meth:`.select_points`
        :meth:`.select_index`

        Notes
        -----

        The returned dataset will most likely not have sufficient coordinate data
        to be used with a particular :class:`Convention` any more.
        The ``dataset.ems`` accessor will raise an error if accessed on the new dataset.
        """
        selector = self.selector_for_indexes(indexes, index_dimension=index_dimension)

        # Make a new dataset consisting of only data arrays that use at least
        # one of these dimensions.
        if drop_geometry:
            dataset = self.drop_geometry()
        else:
            dataset = self.dataset

        dims = set(selector.variables.keys())
        names = [
            name for name, data_array in dataset.items()
            if dims.intersection(data_array.dims)
        ]
        dataset = utils.extract_vars(dataset, names)

        # Select just this point
        return dataset.isel(selector)

    def select_point(self, point: Point) -> xarray.Dataset:
        """
        Return a new dataset that contains values for a single point.
        This is a shortcut for :meth:`get_index_for_point` and :meth:`select_index`.

        If the point is not inside the dataset a :class:`ValueError` is raised.

        Parameters
        ----------
        point : shapely.Point
            The point to select

        Returns
        -------
        xarray.Dataset
            A dataset of values at the point

        See also
        --------
        :meth:`.select_index`
        :meth:`.select_points`
        """
        index = self.get_index_for_point(point)
        if index is None:
            raise ValueError("Point did not intersect dataset")
        return self.select_index(index.index)

    def select_points(
        self,
        points: list[Point],
        *,
        point_dimension: Hashable | None = None,
        missing_points: Literal['error', 'drop'] = 'error',
    ) -> xarray.Dataset:
        """
        Extract values from all variables on the default grid at a sequence of points.

        Parameters
        ----------
        points : list of shapely.Point
            The points to extract
        point_dimension : str, optional
            The name of the new dimension used to index points.
            Defaults to 'point', or 'point_0', 'point_1', etc if those dimensions already exist.
        missing_points : {'error', 'drop'}, default 'error'
            What to do if a point does not intersect the dataset.
            'raise' will raise an error, while 'drop' will drop those points.

        Returns
        -------
        xarray.Dataset
            A dataset with values extracted from the points.
            No variables not defined on the default grid and no geometry variables will be present.

        See also
        --------
        :meth:`.select_indexes`
        :meth:`.select_point`
        """
        if point_dimension is None:
            point_dimension = utils.find_unused_dimension(self.dataset, 'point')
        return point_extraction.extract_points(
            self.dataset, points, point_dimension=point_dimension, missing_points=missing_points)

    @abc.abstractmethod
    def get_all_geometry_names(self) -> list[Hashable]:
        """
        Return a list of the names of all geometry variables used by this convention.

        See Also
        --------
        drop_geometry
        select_variables
        """
        pass

    def drop_geometry(self) -> xarray.Dataset:
        """
        Return a new :class:`xarray.Dataset`
        with all geometry variables dropped.
        Useful when significantly transforming the dataset,
        such as :mod:`extracting point data <emsarray.operations.point_extraction>`.

        See Also
        --------
        get_all_geometry_names
        select_variables
        """
        return self.dataset.drop_vars(self.get_all_geometry_names())

    def select_variables(self, variables: Iterable[DataArrayOrName]) -> xarray.Dataset:
        """Select only a subset of the variables in this dataset, dropping all others.

        This will keep all coordinate variables and all geometry variables.

        Parameters
        ----------
        variables : iterable of DataArrayOrName
            The data variables to select.

        Returns
        -------
        xarray.DataArray
            A new dataset with the same geometry and coordinates,
            but only the selected data variables.

        See also
        --------
        get_all_geometry_names
        drop_geometry
        """
        all_vars = set(self.dataset.variables.keys())
        keep_vars = [
            *variables,
            *self.get_all_geometry_names(),
            *self.depth_coordinates,
        ]
        try:
            keep_vars.append(self.time_coordinate)
        except NoSuchCoordinateError:
            pass
        keep_var_names = {utils.data_array_to_name(self.dataset, v) for v in keep_vars}
        return self.dataset.drop_vars(all_vars - keep_var_names)

    @abc.abstractmethod
    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        *,
        buffer: int = 0,
    ) -> xarray.Dataset:
        """
        Make a new Dataset that can be used to clip this dataset to only the
        cells that intersect some geometry.

        This dataset can be saved to a file to be reused to cut multiple
        datasets with identical shapes, such as a series of files representing
        multiple time series of a model.

        The mask can be applied to this dataset (or other datasets identical in
        shape) using :meth:`apply_clip_mask`.

        Parameters
        ----------
        clip_geometry : shapely.BaseGeometry
            The desired area to cut out. This can be any shapely geometry type,
            but will most likely be a polygon
        buffer : int, optional
            If set to a positive integer,
            a buffer of that many cells will be added around the clip region.
            This is useful if you need to clip to a particular area,
            but also would like to do some interpolation on the output cells.

        Returns
        -------
        :class:`xarray.Dataset`
            The mask

        See Also
        --------
        :func:`apply_clip_mask`
        :func:`clip`
        """
        pass

    @abc.abstractmethod
    def apply_clip_mask(self, clip_mask: xarray.Dataset, work_dir: Pathish) -> xarray.Dataset:
        """
        Apply a clip mask to this dataset, and return a new dataset.
        Call :func:`make_clip_mask` to create a clip mask from a clip geometry.

        The ``clip_mask`` can be saved and loaded to disk if the mask needs to
        be reused across multiple datasets, such as multiple time series from
        one model.

        Depending on the implementation, the input dataset may be sliced in to
        multiple files during cutting, and the returned :class:`~xarray.Dataset`
        might be a multi-file :class:`~xarray.Dataset` built from these
        temporary files. The caller must either load the dataset in to memory
        using :meth:`~xarray.Dataset.load` or :meth:`~xarray.Dataset.compute`,
        or save the dataset to disk somewhere outside of the working directory
        before the working directory is cleaned up.

        Parameters
        ----------
        clip_mask : xarray.Dataset
            The mask, as made by :meth:`make_clip_mask`.
        work_dir : str or pathlib.Path
            A directory where temporary files can be written to.
            Callers must create and manage this temporary directory,
            perhaps using :obj:`tempfile.TemporaryDirectory`.

        Returns
        -------
        xarray.Dataset
            A new :class:`~xarray.Dataset` clipped using the mask
        """

    def clip(
        self,
        clip_geomery: BaseGeometry,
        work_dir: Pathish,
        *,
        buffer: int = 0,
    ) -> xarray.Dataset:
        """
        Generates a clip mask and applies it in one step.

        See the documentation for :meth:`.make_clip_mask` and
        :meth:`.apply_clip_mask` for more details.

        Parameters
        ----------
        clip_geometry : shapely.BaseGeometry
            The desired area to cut out.
            This can be any shapely geometry type,
            but will most likely be a polygon
        work_dir : str or pathlib.Path
            A directory where temporary files can be written to.
            Callers must create and manage this temporary directory,
            perhaps using :obj:`tempfile.TemporaryDirectory`.
        buffer : int, optional
            If set to a positive integer,
            a buffer of that many cells will be added around the clip region.
            This is useful if you need to clip to a particular area,
            but also would like to do some interpolation on the output cells.

        Returns
        -------
        xarray.Dataset
            A new :class:`~xarray.Dataset` clipped using the mask
        """
        mask = self.make_clip_mask(clip_geomery, buffer=buffer)
        return self.apply_clip_mask(mask, work_dir=work_dir)

    def to_netcdf(self, path: Pathish, **kwargs: Any) -> None:
        """
        Save this dataset to a netCDF file, and also fix up the time units to
        make the EMS compatible.
        """
        try:
            time_variable = self.time_coordinate
        except KeyError:
            time_variable = None
        utils.to_netcdf_with_fixes(
            self.dataset, path, time_variable=time_variable, **kwargs)

    # Aliases for emsarray.operations

    def ocean_floor(self) -> xarray.Dataset:
        """An alias for :func:`emsarray.operations.depth.ocean_floor`"""
        return depth.ocean_floor(
            self.dataset, self.depth_coordinates,
            non_spatial_variables=[self.time_coordinate])

    def normalize_depth_variables(
        self,
        *,
        positive_down: bool | None = None,
        deep_to_shallow: bool | None = None,
    ) -> xarray.Dataset:
        """An alias for :func:`emsarray.operations.depth.normalize_depth_variables`"""
        return depth.normalize_depth_variables(
            self.dataset, self.depth_coordinates,
            positive_down=positive_down, deep_to_shallow=deep_to_shallow)


class DimensionConvention(Convention[GridKind, Index]):
    """
    A Convention subclass where different grid kinds
    are always defined on unique sets of dimension.
    This covers most conventions.

    This subclass adds the abstract methods and properties:

    - :attr:`.grid_dimensions`
    - :meth:`.unpack_index`
    - :meth:`.pack_index`

    Default implementations are provided for:

    - :attr:`.grid_size`
    - :meth:`.get_grid_kind`
    - :meth:`.ravel_index`
    - :meth:`.wind_index`
    - :meth:`.ravel`
    - :meth:`.wind`
    - :meth:`.selector_for_index`
    """

    @property
    @abc.abstractmethod
    def grid_dimensions(self) -> dict[GridKind, Sequence[Hashable]]:
        """
        The dimensions associated with a particular grid kind.

        This is a mapping between :data:`grid kinds <GridKind>`
        and an ordered list of dimension names.
        Each dimension in the dataset must be associated with at most one grid kind.
        Each grid kind must be associated with at least one dimension.
        The dimensions must be in the order expected in a dataset,
        if order is significant.

        This property may introspect the dataset
        to determine which dimensions are used.
        The property should be cached.
        """
        pass

    @property
    def grid_shape(self) -> dict[GridKind, Sequence[int]]:
        """
        The :attr:`shape <numpy.ndarray.shape>` of each grid kind.

        :meth private:
        """
        return {
            grid_kind: tuple(
                self.dataset.sizes[dim]
                for dim in self.grid_dimensions[grid_kind]
            )
            for grid_kind in self.grid_kinds
        }

    @property
    def grid_size(self) -> dict[GridKind, int]:
        return {
            grid_kind: int(numpy.prod(shape))
            for grid_kind, shape in self.grid_shape.items()
        }

    def get_grid_kind(self, data_array: xarray.DataArray) -> GridKind:
        actual_dimensions = set(data_array.dims)
        for kind, dimensions in self.grid_dimensions.items():
            if actual_dimensions.issuperset(dimensions):
                return kind
        raise ValueError("Unknown grid kind")

    @abc.abstractmethod
    def unpack_index(self, index: Index) -> tuple[GridKind, Sequence[int]]:
        """Convert a native index in to a grid kind and dimension indexes.

        Parameters
        ----------
        index : Index
            A native index

        Returns
        -------
        grid_kind : GridKind
            The grid kind
        indexes : sequence of int
            The dimension indexes

        See Also
        --------
        pack_index
        """

        pass

    @abc.abstractmethod
    def pack_index(self, grid_kind: GridKind, indexes: Sequence[int]) -> Index:
        """Convert a grid kind and dimension indexes in to a native index.

        Parameters
        ----------
        grid_kind : GridKind
            The grid kind
        indexes : sequence of int
            The dimension indexes

        Returns
        -------
        index : Index
            The corresponding native index

        See Also
        --------
        unpack_index
        """
        pass

    def ravel_index(self, index: Index) -> int:
        grid_kind, indexes = self.unpack_index(index)
        shape = self.grid_shape[grid_kind]
        return int(numpy.ravel_multi_index(indexes, shape))

    def wind_index(
        self,
        linear_index: int,
        *,
        grid_kind: GridKind | None = None,
    ) -> Index:
        if grid_kind is None:
            grid_kind = self.default_grid_kind
        shape = self.grid_shape[grid_kind]
        indexes = tuple(map(int, numpy.unravel_index(linear_index, shape)))
        return self.pack_index(grid_kind, indexes)

    def ravel(
        self,
        data_array: xarray.DataArray,
        *,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        kind = self.get_grid_kind(data_array)
        dimensions = self.grid_dimensions[kind]
        return utils.ravel_dimensions(
            data_array, list(dimensions),
            linear_dimension=linear_dimension)

    def wind(
        self,
        data_array: xarray.DataArray,
        *,
        grid_kind: GridKind | None = None,
        axis: int | None = None,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        if axis is not None:
            linear_dimension = data_array.dims[axis]
        elif linear_dimension is None:
            linear_dimension = data_array.dims[-1]
        if grid_kind is None:
            grid_kind = self.default_grid_kind

        dimensions = self.grid_dimensions[grid_kind]
        sizes = [self.dataset.sizes[dim] for dim in dimensions]

        return utils.wind_dimension(
            data_array,
            dimensions=dimensions, sizes=sizes,
            linear_dimension=linear_dimension)

    def selector_for_indexes(
        self,
        indexes: list[Index],
        *,
        index_dimension: Hashable | None = None,
    ) -> xarray.Dataset:
        if index_dimension is None:
            index_dimension = utils.find_unused_dimension(self.dataset, 'index')
        if len(indexes) == 0:
            raise ValueError("Need at least one index to select")

        grid_kinds, index_tuples = zip(*[self.unpack_index(index) for index in indexes])

        unique_grid_kinds = set(grid_kinds)
        if len(unique_grid_kinds) > 1:
            raise ValueError(
                "All indexes must be on the same grid kind, got "
                + ", ".join(map(repr, unique_grid_kinds)))

        grid_kind = grid_kinds[0]
        dimensions = self.grid_dimensions[grid_kind]
        # This array will have shape (len(indexes), len(dimensions))
        index_array = numpy.array(index_tuples)
        return xarray.Dataset({
            dimension: (index_dimension, index_array[:, i])
            for i, dimension in enumerate(dimensions)
        })
