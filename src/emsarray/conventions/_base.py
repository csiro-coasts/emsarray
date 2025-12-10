import abc
import dataclasses
import enum
import hashlib
import logging
import math
import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy
import shapely
import xarray
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from emsarray import plot as _plot
from emsarray import utils
from emsarray.exceptions import InvalidGeometryWarning, NoSuchCoordinateError
from emsarray.operations import depth, point_extraction, triangulate
from emsarray.operations.cache import hash_attributes, hash_int, hash_string
from emsarray.state import State
from emsarray.types import Bounds, DataArrayOrName, Pathish

if TYPE_CHECKING:
    # Import these optional dependencies only during type checking
    from cartopy.crs import CRS
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from emsarray.plot import GridArtist

logger = logging.getLogger(__name__)


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


@dataclasses.dataclass
class Grid[GridKind, Index](abc.ABC):
    #: The :class:`Convention` this grid is associated with
    convention: 'Convention[GridKind, Index]'

    #: The :type:`GridKind` this grid represents
    grid_kind: GridKind

    #: A Shapely geometry class such as :class:`shapely.Polygon` or :class:`shapely.Point`.
    geometry_type: type[BaseGeometry]

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        The linear :attr:`~xarray.DataArray.size`
        of :class:`DataArrays <xarray.DataArray>` on this grid,
        not accounting for other dimensions such as time or depth.
        """
        pass

    @cached_property
    def geometry(self) -> numpy.ndarray:
        """
        The geometry of this grid as a :class:`numpy.ndarray` of Shapely geometries.
        Different grids might have different :attr:`geometry types <.geometry_type>`.
        Some elements may be `None`, depending on the dataset convention and
        the particulars of the dataset.

        See also
        --------
        :attr:`Grid.geometry_type`
        :attr:`Grid.mask`
        """
        return self.convention.make_geometry(self.grid_kind)

    @cached_property
    def strtree(self) -> STRtree:
        """
        A :class:`shapely.strtree.STRtree` filled with the geometry of this grid.
        The indexes returned when querying this STRtree correspond to the linear index of this grid.
        The indexes correspond with the geometry in :attr:`Grid.geometry`.

        Example
        -------

        This example will find the linear index, native index, selector, and geometry
        at a point.

        .. code-block:: python

            import emsarray
            import shapely

            dataset = emsarray.tutorial.open_dataset('gbr4')
            point = shapely.Point(151.869, -23.386)
            grid = dataset.ems.grids['face']
            intersecting_indexes = grid.strtree.query(point, predicate="intersects")

            if len(intersecting_indexes) == 0:
                print("No intersecting geometry")

            else:
                linear_index = intersecting_indexes[0]
                native_index = grid.wind_index(linear_index)
                selector = dataset.ems.selector_for_index(native_index)
                geometry = grid.geometry[linear_index]

        See also
        --------
        :attr:`Grid.geometry`
        :meth:`Grid.wind_index()`
        """
        return STRtree(self.geometry)

    @cached_property
    def mask(self) -> numpy.ndarray:
        """
        A boolean :class:`numpy.ndarray` of the same length as :attr:`Grid.geometry`,
        True where the geometry is defined, False if the geometry is undefined or invalid.
        """
        mask = numpy.fromiter(
            (p is not None for p in self.geometry),
            dtype=bool, count=self.size)
        return cast(numpy.ndarray, mask)

    @cached_property
    def centroid(self) -> numpy.ndarray:
        """
        The centres of the geometry of this grid as a :class:`numpy.ndarray` of Shapely points.
        Defaults to the :func:`shapely.centroid` of :attr:`Grid.geometry`,
        but some conventions might have more specific ways of finding the centres.
        """
        return self.convention.make_geometry_centroid(self.grid_kind)

    @abc.abstractmethod
    def ravel_index(self, index: Index) -> int:
        """
        Convert a native :type:`Index` to a linear index for this grid.
        """
        pass

    @abc.abstractmethod
    def wind_index(self, linear_index: int) -> Index:
        """
        Convert a linear index to a native :type:`Index` for this grid.
        """
        pass

    @abc.abstractmethod
    def ravel(
        self,
        data_array: DataArrayOrName,
        *,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        """
        Flatten the surface dimensions of a :class:`~xarray.DataArray`,
        returning a flatter :class:`numpy.ndarray` indexed in the same order as the linear index.

        For DataArrays with extra dimensions such as time or depth,
        only the surface dimensions are flattened.
        Other dimensions are left as is.

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
            A new data array where all the surface dimensions
            have been flattened in to one linear array.
            The values for each cell are in the same order as the linear index for this grid.
            Any other dimensions, such as depth or time, will be retained.

        See Also
        --------
        Convention.ravel : An alias for this that will find the correct grid for a data array first.
        .utils.ravel_dimensions : A function that ravels some given dimensions in a dataset.
        """
        pass

    @abc.abstractmethod
    def wind(
        self,
        data_array: xarray.DataArray,
        *,
        axis: int | None = None,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        """
        Wind a flattened :class:`~xarray.DataArray`
        so that it has the same shape as this grid.

        By using :attr:`.size` and :meth:`.wind()` together
        it is possible to construct new data variables for a dataset
        of any arbitrary shape.

        Parameters
        ----------
        data_array : xarray.DataArray
            One of the data variables from this dataset.
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
            to match the shape of this grid.
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
            grid = dataset.ems.default_grid
            flat_array = xarray.DataArray(
                data=numpy.arange(grid.size),
                dims=['index'],
            )
            data_array = grid.wind(flat_array)

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

            grid = dataset.ems.default_grid
            hits = grid.strtree.query(target, predicate='intersects')
            intersecting_cells = xarray.DataArray(
                data=numpy.zeros(grid.size, dtype=bool),
                dims=['index'],
            )
            intersecting_cells.values[hits] = True
            intersecting_cells = grid.wind(intersecting_cells)

            dataset.ems.plot(intersecting_cells)

        See Also
        --------
        Grid.ravel : The inverse operation.
        Convention.wind : A shortcut to this method on the Convention
        .utils.wind_dimension : Reshape a particular dimension in a data array.
        """
        pass

    def __repr__(self) -> str:
        return f'<Grid: {self.grid_kind} {self.size}>'


class Convention[GridKind, Index](abc.ABC):
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
        xarray will find all time variables and convert them to numpy datetimes when opening a dataset.

        The CF Conventions state that
        a time variable is defined by having a `units` attribute
        formatted according to the UDUNITS package [1]_.

        In practice, some datasets do not follow this conventions.
        emsarray will first look for a time coordinate that conforms to the CF Conventions,
        but if none are found it will fall back to picking a datetime variable
        that has at least one `coordinate_type`: `time`, `standard_name`: `time`, or `axis`: `T` attribute set.

        References
        ----------
        .. [1] `CF Conventions v1.10, 4.4 Time Coordinate <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#time-coordinate>`_
        """
        # First look for a datetime64 variable with a 'units' field in the encoding
        for name in self.dataset.variables.keys():
            variable = self.dataset[name]
            # The variable must be a numpy datetime
            if variable.dtype.type == numpy.datetime64:
                # xarray will automatically decode all time variables
                # and move the 'units' attribute over to encoding to store this change.
                if 'units' in variable.encoding:
                    units = variable.encoding['units']
                    # A time variable must have units of the form '<units> since <epoc>'
                    if 'since' in units:
                        return variable

        # Next, look for any datetime64 variable with an appropriate attribute
        for name in self.dataset.variables.keys():
            variable = self.dataset[name]
            if variable.dtype.type == numpy.datetime64:
                possible_attributes = {
                    'coordinate_type': 'time',
                    'standard_name': 'time',
                    'axis': 'T',
                }
                if any(variable.attrs.get(name, None) == value for name, value in possible_attributes.items()):
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

    def ravel_index(self, index: Index) -> int:
        """Convert a convention native index to a linear index.

        Each convention has a different native index type,
        read the specific convention documentation for more information.

        Parameters
        ----------
        index : :type:`.Index`
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
            >>> dataset.ems.ravel_index((CFGridKind.face, 3, 4))
            124

        Cell polygons are indexed in the same order as the linear indexes for cells.
        To find the polygon for the cell with the native index ``(3, 4)``:

        .. code-block:: python

            >>> grid = dataset.ems.grids[CFGridKind.face]
            >>> index = (CFGridKind.face, 3, 4)
            >>> linear_index = grid.ravel_index(index)
            >>> polygon = grid.geometry[linear_index]

        See Also
        --------
        :meth:`.Convention.wind_index` : The inverse operation
        """
        grid = self.get_grid_for_index(index)
        return grid.ravel_index(index)

    def wind_index(
        self,
        linear_index: int,
        *,
        grid_kind: GridKind | None = None,
    ) -> Index:
        """Convert a linear index to a convention native index.

        Each convention has a different native index type,
        read the specific convention documentation for more information.

        Parameters
        ----------
        linear_index : int
            The linear index to wind.
        grid_kind : :type:`.GridKind`, optional
            Used to indicate what kind of index is being wound,
            for conventions with multiple grids.
            Optional, if not provided the default grid kind will be used.

        Returns
        -------
        :type:`.Index`
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
            (CFGridKind.face, 3, 4)

        See Also
        --------
        :meth:`.Convention.ravel_index` : The inverse operation
        """
        if grid_kind is None:
            grid = self.default_grid
        else:
            grid = self.grids[grid_kind]
        return grid.wind_index(linear_index)

    @property
    @abc.abstractmethod
    def grid_kinds(self) -> frozenset[GridKind]:
        """
        A set of the :type:`grid kinds <.GridKind>` this dataset includes.
        """
        pass

    @property
    @abc.abstractmethod
    def default_grid_kind(self) -> GridKind:
        """
        The default :type:`grid kind <.GridKind>` for this dataset.
        For most datasets this should be a polygon grid.
        """
        # TODO Deprecate this?
        pass

    @property
    @abc.abstractmethod
    def grids(self) -> dict[GridKind, Grid[GridKind, Index]]:
        """
        All of the grids this dataset is defined on,
        as a dict of :type:`GridKind`: :class:`Grid`.
        Every :type:`GridKind` in :attr:`Convention.grid_kinds`
        is present as a key in the dictionary.
        """
        pass

    @cached_property
    def default_grid(self) -> Grid[GridKind, Index]:
        return self.grids[self.default_grid_kind]

    @cached_property
    @utils.deprecated(
        "Convention.grid_size[grid_kind] is deprecated, "
        "use Convention.grids[grid_kind].size instead."
    )
    def grid_size(self) -> dict[GridKind, int]:
        """
        The linear size of each grid kind.

        .. deprecated:: 1.0.0

            Use :attr:`Grid.size` instead:

            .. code-block:: python

                dataset = xarray.open_dataset(...)
                grid = dataset.ems.default_grid
                grid.size
        """
        return {grid_kind: grid.size for grid_kind, grid in self.grids.items()}

    @abc.abstractmethod
    def get_grid_kind(self, data_array: xarray.DataArray) -> GridKind:
        """
        Determines the relevant grid kind for this data array.
        If the data array doesn't match any grid kind for this dataset
        a ValueError is raised

        Parameters
        ----------
        data_array : xarray.DataArray
            The data array to inspect

        Returns
        -------
        :type:`.GridKind`

        Raises
        ------
        ValueError
            If the data array passed in doesn't match any grid kind for this dataset
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

        See also
        --------
        :meth:`Convention.get_grid`
        :attr:`Convention.grid_kinds`
        """
        pass

    def get_grid(self, data_array: xarray.DataArray) -> Grid:
        """
        Get the :class:`Grid` for a :class:`xarray.DataArray`.
        If the data array doesn't match any grid kind for this dataset
        a ValueError is raised

        Parameters
        ----------
        data_array : xarray.DataArray
            A :class:`xarray.DataArray` from this dataset,
            or a data array with matching dimensions.

        Returns
        -------
        Grid
            A :class:`Grid` from :attr:`Convention.grids`

        Raises
        ------
        ValueError
            If the data array passed in doesn't match any grid kind for this dataset
            a ValueError is raised.
            Depth coordinates or time coordinates are examples of data arrays
            that will not be indexable and will raise an error.

        See also
        --------
        :meth:`Convention.get_grid_kind`
        :attr:`Convention.grids`
        """
        return self.grids[self.get_grid_kind(data_array)]

    @abc.abstractmethod
    def get_grid_kind_for_index(self, index: Index) -> GridKind:
        """
        Determines the relevant :type:`GridKind` for this :type:`Index`.

        Parameters
        ----------
        index : Index
            An index

        Returns
        -------
        grid_kind : GridKind
            The GridKind for the Index

        See also
        --------
        get_grid_for_index - Get the grid for an index
        get_grid_kind - Get the grid kind for a data array
        """
        pass

    def get_grid_for_index(self, index: Index) -> Grid[GridKind, Index]:
        """
        Determines the relevant :class:`Grid` for this :type:`Index`.

        Parameters
        ----------
        index : Index
            An index

        Returns
        -------
        grid : Grid
            The Grid for the Index

        See also
        --------
        get_grid_kind_for_index - Get the grid kind for an index
        get_grid - Get the grid for a data array
        """
        return self.grids[self.get_grid_kind_for_index(index)]

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
        Grid.ravel : The actual implementation
        Convention.wind : The inverse operation.
        .utils.ravel_dimensions : A function that ravels some given dimensions in a dataset.
        """
        return self.get_grid(data_array).ravel(data_array)

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

        See :meth:`Grid.wind()` for full documentation.

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

        See Also
        --------
        Grid.wind : The actual implementation.
        .utils.wind_dimension : Reshape a particular dimension in a data array.
        Convention.ravel : The inverse operation.
        """
        if grid_kind is None:
            grid_kind = self.default_grid_kind
        return self.grids[grid_kind].wind(data_array, axis=axis, linear_dimension=linear_dimension)

    @cached_property  # type: ignore
    @_plot._requires_plot
    def data_crs(self) -> 'CRS':
        """
        The coordinate reference system that coordinates in this dataset are
        defined in.
        Used by plotting methods when creating :class:`Artists <matplotlib.artist.Artist>`.
        Defaults to :class:`cartopy.crs.PlateCarree`.

        If your dataset uses a different coordinate reference system
        this property can be set manually:

        .. code-block:: python

            dataset = emsarray.open_dataset(...)
            dataset.ems.data_crs = pyproj.CRS('EPSG:32755')
        """
        # Lazily imported here as cartopy is an optional dependency
        from cartopy.crs import PlateCarree
        return PlateCarree()

    @_plot._requires_plot
    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Plot a data array and automatically display it.

        This method is most useful when working in Jupyter notebooks
        which display figures automatically.
        This method is a wrapper around :meth:`.plot_on_figure`
        that creates and shows a :class:`~matplotlib.figure.Figure` for you.
        All arguments are passed on to :meth:`.plot_on_figure`,
        refer to that function for details.

        This method is a shortcut for quickly generating simple plots.
        It is not intended to be fully featured.
        See the :ref:`examples <examples>` for more comprehensive plotting examples.

        See Also
        --------
        :meth:`.plot_on_figure`
        """
        from matplotlib import pyplot
        self.plot_on_figure(pyplot.figure(), *args, **kwargs)
        pyplot.show()

    @_plot._requires_plot
    def plot_on_figure(
        self,
        figure: 'Figure',
        *variables: DataArrayOrName | tuple[DataArrayOrName, ...],
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

        This method is a shortcut for quickly generating simple plots.
        It is not intended to be fully featured.
        See the :ref:`examples <examples>` for more comprehensive plotting examples.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The :class:`~matplotlib.figure.Figure` instance to plot this on.
        *variables : :class:`xarray.DataArray` or tuples of :class:`xarray.DataArray`
            Any number of dataset variables to plot.
            Scalar variables should be passed in directly,
            while vector pairs should be passed in as a tuple.
            These will be passed to :meth:`.Convention.make_artist`.
        scalar : DataArrayOrName
            The :class:`~xarray.DataArray` to plot,
            or the name of an existing DataArray in this Dataset.

            .. deprecated:: 1.0.0

                Pass in variables as positional arguments instead
        vector : tuple of DataArrayOrName
            A tuple of the *u* and *v* components of a vector.
            The components should be a :class:`~xarray.DataArray`,
            or the name of an existing DataArray in this Dataset.

            .. deprecated:: 1.0.0

                Pass in variables as positional arguments instead
        **kwargs
            Any extra keyword arguments are passed on to
            :meth:`emsarray.plot.plot_on_figure`

        See Also
        --------
        :meth:`Convention.plot` : A shortcut that automatically displays the figure
        :func:`.plot.plot_on_figure` : The underlying implementation
        """
        if scalar is not None:
            warnings.warn(
                (
                    "The 'scalar' parameter to 'Convention.plot_on_figure() is deprecated. "
                    "Pass the scalar variable as a positional argument instead."
                ),
                category=DeprecationWarning, stacklevel=2)
            variables = variables + (scalar,)

        if vector is not None:
            warnings.warn(
                (
                    "The 'vector' parameter to 'Convention.plot_on_figure() is deprecated. "
                    "Pass the vector tuple as a positional argument instead."
                ),
                category=DeprecationWarning, stacklevel=2)
            variables = variables + (vector,)

        mapped_variables = [
            utils.names_to_data_arrays(self.dataset, v)
            for v in variables
        ]

        if title is not None:
            kwargs['title'] = title

        # Find a title if there is a single variable passed in
        elif len(mapped_variables) == 1 and isinstance(mapped_variables[0], xarray.DataArray):
            variable = mapped_variables[0]
            kwargs['title'] = _plot.make_plot_title(self.dataset, variable)

        _plot.plot_on_figure(figure, self, *mapped_variables, **kwargs)

    @_plot._requires_plot
    def animate_on_figure(
        self,
        figure: 'Figure',
        *variables: DataArrayOrName | tuple[DataArrayOrName, ...],
        scalar: DataArrayOrName | None = None,
        vector: tuple[DataArrayOrName, DataArrayOrName] | None = None,
        coordinate: DataArrayOrName | None = None,
        title: str | Callable[[Any], str] | None = None,
        **kwargs: Any,
    ) -> 'FuncAnimation':
        """
        Make an animated plot of a data array.

        This method is a shortcut for quickly generating simple animations.
        It is not intended to be fully featured.
        See the :ref:`examples <examples>` for more comprehensive plotting examples.

        For real world examples, refer to the ``examples/animation.ipynb`` notebook.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The :class:`matplotlib.figure.Figure` to plot the animation on
        *variables : :class:`xarray.DataArray` or tuples of :class:`xarray.DataArray`
            Any number of dataset variables to plot.
            Scalar variables should be passed in directly,
            while vector pairs should be passed in as a tuple.
            These will be passed to :meth:`.Convention.make_artist`.
        scalar : DataArrayOrName
            The :class:`~xarray.DataArray` to plot,
            or the name of an existing DataArray in this Dataset.

            .. deprecated:: 1.0.0

                Pass in variables as positional arguments instead
        vector : tuple of DataArrayOrName
            A tuple of the *u* and *v* components of a vector.
            The components should be a :class:`~xarray.DataArray`,
            or the name of an existing DataArray in this Dataset.

            .. deprecated:: 1.0.0

                Pass in variables as positional arguments instead
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

        if scalar is not None:
            warnings.warn(
                (
                    "The 'scalar' parameter to 'Convention.animate_on_figure() is deprecated. "
                    "Pass the scalar variable as a positional argument instead."
                ),
                category=DeprecationWarning, stacklevel=2)
            variables = variables + (scalar,)

        if vector is not None:
            warnings.warn(
                (
                    "The 'vector' parameter to 'Convention.animate_on_figure() is deprecated. "
                    "Pass the vector tuple as a positional argument instead."
                ),
                category=DeprecationWarning, stacklevel=2)
            variables = variables + (vector,)

        mapped_variables = [
            utils.name_to_data_array(self.dataset, v)
            for v in variables
        ]

        if title is not None:
            kwargs['title'] = title

        elif len(mapped_variables) == 1 and isinstance(mapped_variables[0], xarray.DataArray):
            # Make a title out of the scalar variable, but only if a title
            # hasn't been supplied and we don't also have vectors to plot.
            #
            # We can't make a good name from vectors,
            # as they are in two variables with names like
            # 'u component of current' and 'v component of current'.
            #
            # Users can supply their own titles if this automatic behaviour is insufficient
            variable = mapped_variables[0]
            title_bits = []
            variable_title = _plot.make_plot_title(self.dataset, variable)
            if variable_title is not None:
                title_bits.append(variable_title)
            coordinate_title = _plot.make_plot_title(self.dataset, coordinate)
            if coordinate_title is not None:
                title_bits.append(coordinate_title + ': {}')
            else:
                title_bits.append('{}')

            kwargs['title'] = '\n'.join(title_bits)

        return _plot.animate_on_figure(figure, self, coordinate, *mapped_variables, **kwargs)

    @abc.abstractmethod
    def make_artist(
        self,
        axes: 'Axes',
        variable: DataArrayOrName | tuple[DataArrayOrName, ...],
        **kwargs: Any,
    ) -> '_plot.GridArtist':
        """
        Make a matplotlib artists for the data array,
        adding it to the Axes and returning the Artist.
        This method will pick a reasonable way of drawing the variable
        based on the convention and the grid that the variable is defined on.
        See the documentation on each convention for specific details.

        For most conventions:

        * scalar variables defined on a polygon grid will use
          :func:`~emsarray.plot.make_polygon_scalar_collection`
        * vector pairs defined on a polygon grid will use
          :func:`~emsarray.plot.make_polygon_vector_quiver`
        * scalar variables defined on a node grid will use
          :func:`~emsarray.plot.make_node_scalar_artist`

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes to add the artist to.
        variable : DataArrayOrName or tuple of DataArrayOrName
            The data array, or tuple of data arrays, to make an artist for.
            A sensible artist type is picked based on the data arrays passed in
            and the grids they are defined on.
        kwargs : Any
            Any extra kwargs are passed on to the artist
            and can be used to style it.
            The specific kwargs that are accepted depend on the artist that is used.

        Returns
        -------
        emsarray.plot.GridArtist
            The artist for the data array passed in.
            The artist will already have been added to the axes.

        See also
        --------
        :mod:`emsarray.plot` : For a list of all supported artists
        """
        pass

    @abc.abstractmethod
    def plot_geometry(
        self,
        axes: 'Axes',
    ) -> 'GridArtist':
        """
        Plot the geometry of this dataset on the axes.
        What this means is dependent on the dataset convention used.

        Parameters
        ----------
        axes : matplotlib.axes.Axes

        Returns
        -------
        emsarray.plot.GridArtist
            The artists that will draw the geometry.
        """
        pass

    @_plot._requires_plot
    @utils.timed_func
    @utils.deprecated(
        "Convention.make_poly_collection() is deprecated. "
        "Use Convention.make_artist() or emsarray.plot.make_polygon_scalar_collection instead."
    )
    def make_poly_collection(
        self,
        data_array: DataArrayOrName | None = None,
        **kwargs: Any,
    ) -> 'GridArtist':
        """
        Make a :class:`~matplotlib.collections.PolyCollection`
        from the geometry of this :class:`~xarray.Dataset`.

        .. deprecated:: 1.0.0

            Use :meth:`Convention.make_artist()` instead

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
        """
        grid = self.default_grid

        if data_array is not None:
            if 'array' in kwargs:
                raise TypeError(
                    "Can not pass both `data_array` and `array` to make_poly_collection"
                )

            kwargs['data_array'] = utils.name_to_data_array(self.dataset, data_array)
            if 'clim' not in kwargs:
                kwargs['clim'] = (numpy.nanmin(kwargs['data_array']), numpy.nanmax(kwargs['data_array']))

        if 'transform' not in kwargs:
            kwargs['transform'] = self.data_crs

        return _plot.artists.PolygonScalarCollection(grid=grid, **kwargs)

    @_plot._requires_plot
    @utils.deprecated(
        "Convention.make_quiver() is deprecated. "
        "Use Convention.make_artist() or plot.make_polygon_vector_quiver() instead."
    )
    def make_quiver(
        self,
        axes: 'Axes',
        u: DataArrayOrName | None = None,
        v: DataArrayOrName | None = None,
        **kwargs: Any,
    ) -> 'GridArtist':
        """
        Make a :class:`matplotlib.quiver.Quiver` instance to plot vector data.

        .. deprecated:: 1.0.0

            Use :meth:`Convention.make_artist()` instead.

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
        grid = self.default_grid

        if u is not None and v is not None:
            u = utils.name_to_data_array(self.dataset, u)
            v = utils.name_to_data_array(self.dataset, v)
            kwargs['data_array'] = (u, v)

        if 'transform' not in kwargs:
            kwargs['transform'] = self.data_crs

        return _plot.artists.PolygonVectorQuiver(grid=grid, **kwargs)

    def _validate_geometry(
        self,
        grid_kind: GridKind,
        geometry: numpy.ndarray,
    ) -> numpy.ndarray:
        """Check generated geometry to make sure it is valid."""
        grid = self.grids[grid_kind]
        if grid.size != len(geometry):
            raise RuntimeError("Length of grid geometry did not match grid size")

        not_none = (geometry != None)  # noqa: E711
        invalid_indices = numpy.flatnonzero(not_none & ~shapely.is_valid(geometry))
        if len(invalid_indices):
            indices_str = numpy.array2string(
                invalid_indices, max_line_width=None, threshold=5)
            warnings.warn(
                f"Dropping invalid {grid_kind} geometry at indices {indices_str}",
                category=InvalidGeometryWarning)
            geometry[invalid_indices] = None
            not_none[invalid_indices] = False

        geometry.flags.writeable = False

        return geometry

    @abc.abstractmethod
    def _make_geometry(self, grid_kind: GridKind) -> numpy.ndarray:
        """
        Make the geometry for the given grid kind.
        Called by :attr:`Grid.geometry` lazily when required and then cached.
        Subclasses must implement this.
        """
        pass

    def make_geometry(self, grid_kind: GridKind) -> numpy.ndarray:
        """
        Make geometry for the specified :type:`GridKind`.
        This should normally be accessed via :attr:`Grid.geometry`,
        which stores a cached copy of the geometry for each grid.
        """
        geometry = self._make_geometry(grid_kind)
        self._validate_geometry(grid_kind, geometry)
        return geometry

    def _make_geometry_centroid(self, grid_kind: GridKind) -> numpy.ndarray:
        """
        Make geometry centroids for the specified :type:`GridKind`.
        Called by :attr:`Grid.centroid` lazily when required and then cached.
        Subclasses may implement this if this information is part of the convention,
        otherwise falls back to calling :func:`shapely.centroid` on the grid geometry.
        """
        grid = self.grids[grid_kind]
        return cast(numpy.ndarray, shapely.centroid(grid.geometry))

    def make_geometry_centroid(self, grid_kind: GridKind) -> numpy.ndarray:
        """
        Make geometry centroids for the specified :type:`GridKind`.
        This should normally be accessed via :attr:`Grid.centroid`,
        which stores a cached copy of the geometry for each grid.
        """
        centroids = self._make_geometry_centroid(grid_kind)
        self._validate_geometry(grid_kind, centroids)
        return centroids

    @cached_property
    @utils.deprecated(
        "dataset.ems.polygons is deprecated. "
        "Use dataset.ems.get_grid(data_array).geometry instead."
    )
    def polygons(self) -> numpy.ndarray:
        grid = self.grids[self.default_grid_kind]
        if issubclass(grid.geometry_type, Polygon):
            return grid.geometry
        raise ValueError(f"Default grid kind {grid.grid_kind} does not have polygonal geometry")

    @cached_property
    @utils.deprecated(
        "dataset.ems.face_centres is deprecated. "
        "Use dataset.ems.get_grid(data_array).centroid instead. "
        "For a list of coordinate pairs use shapely.get_coordinates(grid.centroid)."
    )
    def face_centres(self) -> numpy.ndarray:
        grid = self.grids[self.default_grid_kind]
        centroid = grid.centroid
        coords = numpy.full(fill_value=numpy.nan, shape=(grid.size, 2))
        coords[centroid != None] = shapely.get_coordinates(centroid)  # noqa: E711
        return cast(numpy.ndarray, coords)

    @cached_property
    @utils.deprecated(
        "dataset.ems.mask is deprecated. "
        "Use dataset.ems.get_grid(data_array).mask instead."
    )
    def mask(self) -> numpy.ndarray:
        grid = self.grids[self.default_grid_kind]
        if issubclass(grid.geometry_type, Polygon):
            return grid.mask
        raise ValueError(f"Default grid kind {grid.grid_kind} does not have polygonal geometry")

    @cached_property
    def geometry(self) -> Polygon | MultiPolygon:
        """
        A :class:`shapely.Polygon` or :class:`shapely.MultiPolygon` that represents
        the geometry of the entire dataset.

        This is equivalent to the union of all polygons in the dataset,
        although specific conventions may have a simpler way of constructing this.
        """
        grid = self.grids[self.default_grid_kind]
        return shapely.unary_union(grid.geometry[grid.mask])

    @cached_property
    def bounds(self) -> Bounds:
        """
        Returns minimum bounding region (minx, miny, maxx, maxy) of the entire dataset.

        This is equivalent to the bounds of the dataset :attr:`geometry`,
        although specific conventions may have a simpler way of constructing this.
        """
        return cast(Bounds, self.geometry.bounds)

    @cached_property
    @utils.deprecated(
        "dataset.ems.strtree is deprecated. "
        "Use dataset.ems.get_grid(data_array).strtree instead."
    )
    def strtree(self) -> STRtree:
        grid = self.grids[self.default_grid_kind]
        return grid.strtree

    def selector_for_index(self, index: Index) -> xarray.Dataset:
        """
        Convert a convention native index into a selector
        that can be passed to :meth:`Dataset.isel <xarray.Dataset.isel>`.

        Parameters
        ----------
        index : :type:`Index`
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
        indexes : list of :type:`Index`
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
        index : :type:`Index`
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
        indexes : list of :type:`Index`
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

    def select_point(
        self,
        point: Point,
        *,
        grid_kind: GridKind | None = None,
    ) -> xarray.Dataset:
        """
        Return a new dataset that contains values for a single point.
        This is a shortcut for querying the :attr:`Grid.strtree` to find intersecting geometry,
        then selecting the cell using :meth:`select_index`.

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
        if grid_kind is None:
            grid_kind = self.default_grid_kind
        grid = self.grids[grid_kind]
        if not issubclass(grid.geometry_type, shapely.Polygon):
            raise ValueError(f"Grid geometry for {grid_kind} is not polygons")

        hits = sorted(grid.strtree.query(point, "intersects"))
        if len(hits) == 0:
            raise ValueError("Point did not intersect dataset")

        index = grid.wind_index(hits[0])
        return self.select_index(index)

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

    def hash_geometry(self, hash: "hashlib._Hash") -> None:
        """
        Updates the provided hash with all of the relevant geometry data for this dataset.

        Parameters
        ----------
        hash : hashlib-style hash instance
            The hash instance to update with geometry data.
            This must follow the interface defined in :mod:`hashlib`.
        """
        geometry_names = self.get_all_geometry_names()

        for geometry_name in geometry_names:
            data_array = self.dataset[geometry_name]

            # Include the variable name in the digest.
            hash_string(hash, str(geometry_name))

            # Include the dtype of the data array.
            # A float array and an int array mean very different things,
            # but could have identical byte patterns.
            # Checking for encoding dtype and falling back to values.dtype due to
            # xarray multifile dataset bug - https://github.com/pydata/xarray/issues/2436
            hash_string(hash, data_array.encoding.get('dtype', data_array.values.dtype).name)

            # Include the size and shape of the data.
            # 1D coordinate arrays are very different to 2D coordinate arrays,
            # but could have identical byte patterns.
            hash_int(hash, data_array.size)
            hash.update(numpy.array(data_array.shape, dtype='int32').tobytes('C'))
            hash.update(data_array.to_numpy().tobytes('C'))

            # Hash dataset attributes
            hash_attributes(hash, data_array.attrs)

    def make_triangulation(self) -> triangulate.Triangulation[GridKind]:
        """
        Triangulates the polygons in the dataset.
        Subclasses may have improved implementations.

        This requires the dataset to have a grid with polygonal geometry.
        If there is an additional grid corresponding to the vertices of the polygons
        then subclasses should use this information to inform the triangulation.

        Returns
        -------
        tuple of vertices, triangles, and `cell_indexes`
            A tuple of three numpy arrays is returned,
            containing vertices, triangles, and cell indexes respectively.

            `vertices` is a numpy array of shape (V, 2)
            where V is the number of unique vertices in the dataset.
            The vertex coordinates are in (x, y) or (lon, lat) order.
            If the dataset has a vertex grid associated with the polygons
            then this array replicates that data.

            `triangles` is a numpy array of shape (T, 3)
            where T is the number of triangles in the dataset.
            Each triangle is a set of three vertex indexes.

            `cell_indexes` is a numpy list of length T.
            Each entry indicates which polygon from the dataset a triangle is a part of.

        See also
        --------
        :func:`~emsarray.operations.triangulate.triangulate`
        """
        grid = self.default_grid
        if not issubclass(grid.geometry_type, shapely.Polygon):
            raise ValueError("Can not triangulate a dataset that does not have polygonal geometry")
        polygons = grid.geometry
        vertices = triangulate.find_unique_vertices(polygons)
        polygon_vertex_indexes = triangulate.polygons_to_vertex_indexes(polygons, vertices)
        vertex_coordinates, triangles, face_indexes = triangulate.triangulate(vertices, polygons, polygon_vertex_indexes)
        return triangulate.Triangulation(
            vertices=vertex_coordinates,
            triangles=triangles,
            face_indexes=face_indexes,
            face_grid_kind=grid.grid_kind)


type DimensionIndex[GridKind] = tuple[GridKind, *tuple[int, ...]]


@dataclasses.dataclass(kw_only=True)
class DimensionGrid[GridKind](Grid[GridKind, DimensionIndex[GridKind]]):
    """
    A :class:`Grid` subclass that complements :class:`DimensionConvention`.
    Has one extra required argument :attr:`dimensions`,
    and provides an implementation of all abstract :class:`Grid` methods.
    """
    #: The dimensions that data arrays on this grid must have.
    dimensions: Sequence[Hashable]

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            self.convention.dataset.sizes[dimension]
            for dimension in self.dimensions
        )

    @cached_property
    def size(self) -> int:
        return int(math.prod(self.shape))

    def ravel_index(self, index: DimensionIndex[GridKind]) -> int:
        grid_kind, *indexes = index
        if grid_kind is not self.grid_kind:
            raise ValueError(
                f"Index is for grid kind {grid_kind}, expected {self.grid_kind}")
        return int(numpy.ravel_multi_index(indexes, self.shape))

    def wind_index(
        self,
        linear_index: int,
    ) -> DimensionIndex[GridKind]:
        indexes = tuple(map(int, numpy.unravel_index(linear_index, self.shape)))
        return (self.grid_kind, *indexes)

    def ravel(
        self,
        data_array: DataArrayOrName,
        *,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        data_array = utils.name_to_data_array(self.convention.dataset, data_array)
        grid_kind = self.convention.get_grid_kind(data_array)
        if grid_kind is not self.grid_kind:
            raise ValueError(
                f"Data array has grid kind {grid_kind}, expected {self.grid_kind}")

        return utils.ravel_dimensions(
            data_array, list(self.dimensions),
            linear_dimension=linear_dimension)

    def wind(
        self,
        data_array: xarray.DataArray,
        *,
        axis: int | None = None,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        if axis is not None:
            linear_dimension = data_array.dims[axis]
        elif linear_dimension is None:
            linear_dimension = data_array.dims[-1]

        return utils.wind_dimension(
            data_array,
            dimensions=self.dimensions, sizes=self.shape,
            linear_dimension=linear_dimension)

    def __repr__(self) -> str:
        return f'<DimensionGrid: {self.grid_kind} {self.dimensions} {self.shape}>'


class DimensionConvention[GridKind](Convention[GridKind, DimensionIndex[GridKind]]):
    """
    A Convention subclass where different grid kinds
    are always defined on unique sets of dimension.
    This covers most conventions.

    This subclass adds the abstract properties
    :attr:`.grid_dimensions` and :attr:`.geometry_types`
    that subclasses must define.

    Default implementations are provided for all abstract :class:`Grid` methods
    using :class:`DimensionGrid`.
    """

    @cached_property
    def grids(self) -> dict[GridKind, Grid[GridKind, DimensionIndex[GridKind]]]:
        return {
            grid_kind: DimensionGrid(
                convention=self,
                grid_kind=grid_kind,
                dimensions=self.grid_dimensions[grid_kind],
                geometry_type=self.geometry_types[grid_kind],
            )
            for grid_kind in self.grid_kinds
        }

    @property
    @abc.abstractmethod
    def grid_dimensions(self) -> dict[GridKind, Sequence[Hashable]]:
        """
        The dimensions associated with a particular grid kind.

        This is a mapping between :type:`grid kinds <GridKind>`
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
    @abc.abstractmethod
    def geometry_types(self) -> dict[GridKind, type[BaseGeometry]]:
        """
        The geometry type associated with a particular grid kind.

        This is a mapping between :type:`grid kinds <GridKind>`
        and shapely geometry types such as
        :class:`~shapely.Polygon` and :class:`~shapely.Point`.

        Users should use :attr:`Grid.geometry_type` instead of this property,
        this is only used to construct the grids.
        """
        pass

    @property
    @utils.deprecated(
        "DimensionConvention.grid_shape[grid_kind] is deprecated. "
        "Use DimensionConvention.grids[grid_kind].shape instead."
    )
    def grid_shape(self) -> dict[GridKind, Sequence[int]]:
        """
        The :attr:`shape <numpy.ndarray.shape>` of each grid kind.

        :meth private:
        """
        return {
            grid_kind: cast(DimensionGrid, grid).shape
            for grid_kind, grid in self.grids.items()
        }

    @property
    @utils.deprecated(
        "DimensionConvention.grid_size[grid_kind] is deprecated. "
        "Use DimensionConvention.grids[grid_kind].size instead."
    )
    def grid_size(self) -> dict[GridKind, int]:
        return {
            grid_kind: grid.size
            for grid_kind, grid in self.grids.items()
        }

    def get_grid_kind(self, data_array: xarray.DataArray) -> GridKind:
        actual_dimensions = set(data_array.dims)
        for kind, dimensions in self.grid_dimensions.items():
            if actual_dimensions.issuperset(dimensions):
                return kind
        raise ValueError("Unknown grid kind")

    def get_grid_kind_for_index(self, index: DimensionIndex[GridKind]) -> GridKind:
        return index[0]

    def selector_for_indexes(
        self,
        indexes: list[DimensionIndex[GridKind]],
        *,
        index_dimension: Hashable | None = None,
    ) -> xarray.Dataset:
        if index_dimension is None:
            index_dimension = utils.find_unused_dimension(self.dataset, 'index')
        if len(indexes) == 0:
            raise ValueError("Need at least one index to select")

        grid_kinds = set(index[0] for index in indexes)
        index_tuples = [index[1:] for index in indexes]

        if len(grid_kinds) > 1:
            raise ValueError(
                "All indexes must be on the same grid kind, got "
                + ", ".join(map(repr, grid_kinds)))

        dimensions = self.grid_dimensions[grid_kinds.pop()]
        # This array will have shape (len(indexes), len(dimensions))
        index_array = numpy.array(index_tuples)
        return xarray.Dataset({
            dimension: (index_dimension, index_array[:, i])
            for i, dimension in enumerate(dimensions)
        })
