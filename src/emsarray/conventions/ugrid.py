"""
Unstructured grid convention.

See Also
--------
`UGRID conventions <https://ugrid-conventions.github.io/ugrid-conventions/>`_
"""
from __future__ import annotations

import enum
import logging
import pathlib
import warnings
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any, Dict, FrozenSet, Hashable, Iterable, List, Mapping, Optional,
    Sequence, Set, Tuple, cast
)

import numpy
import shapely
import xarray
from shapely.geometry.base import BaseGeometry

from emsarray import utils
from emsarray.exceptions import (
    ConventionViolationError, ConventionViolationWarning
)
from emsarray.types import Bounds, Pathish

from ._base import DimensionConvention, Specificity

logger = logging.getLogger(__name__)


def _split_coord(attr: str) -> Tuple[str, str]:
    x, y = attr.split(None, 1)
    return (x, y)


def buffer_faces(face_indices: numpy.ndarray, topology: Mesh2DTopology) -> numpy.ndarray:
    """
    When clipping a dataset to a region, including a buffer of extra faces
    around the included faces is desired. Given an array of face indices,
    :func:`buffer_faces` will find all the faces required to form this buffer
    and return a new array of face indices that include both the input faces
    and the buffer faces

    Specifically, this finds all faces that share a node with one of the
    included faces.
    """
    original_face_indices = set(face_indices.tolist())
    face_node = topology.face_node_array

    # Find all nodes for the faces
    included_nodes = set(numpy.unique(face_node[face_indices].compressed()))

    # Find all faces that are composed of any of the nodes just found
    included_faces = (
        face_index
        for face_index, node_indices in enumerate(face_node)
        # Either one of the original faces ...
        if face_index in original_face_indices
        # ... or shares a node with one of the original faces
        or bool(included_nodes.intersection(node_indices.compressed()))
    )
    return cast(numpy.ndarray, numpy.fromiter(included_faces, dtype=topology.sensible_dtype))


def mask_from_face_indices(face_indices: numpy.ndarray, topology: Mesh2DTopology) -> xarray.Dataset:
    """
    Make a mask dataset from a list of face indices.
    This mask can later be applied using :meth:`~.Convention.apply_clip_mask`.

    A mask for a UGRID dataset indicates which nodes, edges, and faces
    (collectively 'elements') to include, and their new indices. When
    elements are dropped from the dataset, to keep the remaining data
    contiguous, all elements gain new indices.

    The mask consists of three data arrays corresponding to nodes, edges,
    and faces. Each data array indicates the new index for that element, or
    whether that element should be dropped.
    """
    # Generate a new index for each element. Indices are assigned in the same
    # order as they currently exist. This means that data rows for excluded
    # elements can be dropped, and the remaining data rows will automatically
    # have the correct indices.
    fill_value = topology.sensible_fill_value
    data_vars = {}

    def new_element_indices(size: int, indices: numpy.ndarray) -> numpy.ma.MaskedArray:
        new_indices = numpy.full(
            (size,), fill_value=fill_value, dtype=topology.sensible_dtype)
        new_indices = numpy.ma.masked_array(new_indices, mask=True)
        new_indices[indices] = numpy.arange(len(indices))
        return new_indices

    # Record which old face index maps to which new face index
    data_vars['new_face_index'] = _masked_integer_data_array(
        data=new_element_indices(topology.face_count, face_indices),
        fill_value=fill_value,
        dims=['old_face_index'],
    )

    # Find all edges associated with included faces. Don't bother if the
    # dataset doesn't define an edge dimension
    if topology.has_edge_dimension:
        face_edge = topology.face_edge_array
        edge_indices = numpy.sort(numpy.unique(face_edge[face_indices].compressed()))

        # Record which old edge index maps to which new edge index
        data_vars['new_edge_index'] = _masked_integer_data_array(
            data=new_element_indices(topology.edge_count, edge_indices),
            fill_value=fill_value,
            dims=['old_edge_index'],
        )

    # Find all nodes associated with included faces
    face_node = topology.face_node_array
    node_indices = numpy.sort(numpy.unique(face_node[face_indices].compressed()))

    # Record which old node index maps to which new node index
    data_vars['new_node_index'] = _masked_integer_data_array(
        data=new_element_indices(topology.node_count, node_indices),
        fill_value=fill_value,
        dims=['old_node_index'],
    )

    # Make the mask dataset
    return xarray.Dataset(data_vars=data_vars, attrs={
        'title': 'UGRID dataset mask',
    })


def _get_start_index(connectivity: xarray.DataArray) -> int:
    """
    Get the ``start_index`` attribute from a connectivity data array,
    while checking for common error cases.
    """
    # Default to 0 if there is no start_index attribute
    if 'start_index' not in connectivity.attrs:
        return 0

    start_index = connectivity.attrs.get('start_index')

    # These are the valid values
    if start_index in {0, 1}:
        return cast(int, start_index)

    # Some datasets use the strings '0' or '1'.
    # This does not adhere to the spec but is easy to interpret
    if start_index in {'0', '1'}:
        warnings.warn(
            f"Connectivity data array {connectivity.name!r} "
            f"had string-typed start index {start_index!r}. "
            f"This attribute should be the integer {int(start_index)}, "
            f"not a string.",
            ConventionViolationWarning,
        )
        return int(start_index)

    # Anything else is incorrect
    raise ConventionViolationError(
        f"Connectivity data array {connectivity.name!r} "
        f"had invalid start index {start_index!r}"
    )


def update_connectivity(
    connectivity: xarray.DataArray,
    old_array: numpy.ndarray,
    row_indices: numpy.ndarray,
    column_values: numpy.ndarray,
    primary_dimension: Hashable,
    fill_value: int,
) -> xarray.DataArray:
    """
    Create a new connectivity variable by reindexing existing entries.
    This is used during masking to trim off unused nodes, edges, and faces.
    It preserves original dtype, any given ``start_index`` value,
    and non-standard dimension orderings.

    Parameters
    ----------

    connectivity : xarray.DataArray
        The connectivity variable to update.
        This will be one of
        :attr:`~Mesh2DTopology.edge_node_connectivity`,
        :attr:`~Mesh2DTopology.edge_face_connectivity`,
        :attr:`~Mesh2DTopology.face_node_connectivity`,
        :attr:`~Mesh2DTopology.face_edge_connectivity`, or
        :attr:`~Mesh2DTopology.face_face_connectivity`.
    old_array : numpy.ndarray
        The old connectivity array,
        the companion to the connectivity data array.
        Each row corresponds to either an edge or a face.
    row_indices : numpy.ndarray
        A one dimensional numpy masked array
        indicating which rows of ``old_array`` are to be included.
        Masked items are excluded, all other items are included.
    column_values : numpy.ndarray
        A one dimensional numpy masked array
        mapping values from ``old_array`` to their new values.
        Values to be excluded in the new array should be ``numpy.ma.masked``
    primary_dimension : Hashable
        The name of the primary dimension for this connectivity variable.
        This will be either :attr:`Mesh2DTopology.edge_dimension`
        or :attr:`Mesh2DTopology.face_dimension`.
        Each row represents data about either a edge or a face.
    fill_value : int
        The fill value to use for missing data.

    Returns
    -------

    xarray.DataArray
        A new DataArray containing the clipped and reindexed connectivity data.
    """
    logger.debug("Reindexing %r", connectivity.name)

    # Offset the column_values for one-based indexing.
    start_index = _get_start_index(connectivity)
    if start_index != 0:
        column_values = column_values + start_index

    dtype = connectivity.encoding.get('dtype', connectivity.dtype)

    if dtype.kind == 'i':
        # Ensure the fill value fits within the representable integers
        max_representable = numpy.iinfo(dtype).max
        if max_representable < fill_value:
            fill_value = max_representable

    # We need to preseve the integer dtype,
    # while also accounting for masked values.
    # xarray does not make this easy.
    # By constructing the array using new_fill_value where needed,
    # setting the dtype explicitly, and adding the _FillValue attribute,
    # xarray will cooperate.
    include_row = ~numpy.ma.getmask(row_indices)
    raw_values = numpy.array([
        [
            column_values[item] if item is not numpy.ma.masked else fill_value
            for item in row
        ]
        for row in old_array[include_row]
    ], dtype=dtype)
    values = numpy.ma.masked_equal(raw_values, fill_value)

    if connectivity.dims[1] == primary_dimension:
        values = numpy.transpose(values)
    elif primary_dimension not in connectivity.dims:
        raise ValueError("Connectivity variable does not contain primary dimension")

    return _masked_integer_data_array(
        data=values,
        fill_value=fill_value,
        dims=connectivity.dims,
        name=connectivity.name,
        attrs=connectivity.attrs,
    )


def _masked_integer_data_array(
    data: numpy.ma.MaskedArray,
    fill_value: int,
    **kwargs: Any,
) -> xarray.DataArray:
    """
    Create an :class:`xarray.DataArray` that represents
    an integer variable with a _FillValue.
    When reading such a variable from disk,
    xarray will cast it to a double
    and replace any occurrence of _FillValue with numpy.nan.
    This method replicates these actions.
    When saved to disk, a variable built using this should have
    an integer type and a _FillValue.
    """
    float_data = numpy.ma.filled(data.astype(numpy.double), numpy.nan)
    data_array = xarray.DataArray(data=float_data, **kwargs)
    data_array.encoding.update({'dtype': data.dtype, '_FillValue': fill_value})
    return data_array


class NoEdgeDimensionException(ValueError):
    """
    Raised when the dataset does not define a name for the edge dimension, and
    no dimension name can be derived from the available connectivity variables.
    """


class NoConnectivityVariableException(Exception):
    """
    Raised when the dataset does not have a particular connectivity variable
    defined.
    """


@dataclass
class Mesh2DTopology:
    """
    A helper for extracting the topology data for a 2D unstructured mesh
    dataset. The dataset will contain a dummy variable with the attribute
    ``cf_role = "mesh_topology"``. The attributes of that variable explain the
    topology of the dataset. For example, the names of the variables that store
    the coordinates for nodes can be found in the ``node_coordinates`` attribute
    as the string ``"x_variable y_variable"``.

    There are five possible connectivity variables on a UGRID dataset:
    face-node, face-edge, face-face, edge-node, and edge-face.
    Face-node is the only required variable.
    Each has a property to check whether it is defined (``has_valid_foo``),
    a property to get the variable if it is defined (``foo_connectivity``),
    and a property to get or derive a normalised connectivity array (``foo_array``):

    .. list-table::
        :header-rows: 1

        * - Attribute
          - has_valid_foo
          - foo_connectivity
          - foo_array
        * - face_node
          - :attr:`has_valid_face_node_connectivity`
          - :attr:`face_node_connectivity`
          - :attr:`face_node_array`
        * - face_edge
          - :attr:`has_valid_face_edge_connectivity`
          - :attr:`face_edge_connectivity`
          - :attr:`face_edge_array`
        * - face_face
          - :attr:`has_valid_face_face_connectivity`
          - :attr:`face_face_connectivity`
          - :attr:`face_face_array`
        * - edge_node
          - :attr:`has_valid_edge_node_connectivity`
          - :attr:`edge_node_connectivity`
          - :attr:`edge_node_array`
        * - edge_face
          - :attr:`has_valid_edge_face_connectivity`
          - :attr:`edge_face_connectivity`
          - :attr:`edge_face_array`
    """
    #: The UGRID dataset
    dataset: xarray.Dataset

    #: The name of the mesh topology variable. Optional. If not provided, the
    #: mesh topology dummy variable will be found by checking the ``cf_role``
    #: attribute.
    topology_key: Optional[Hashable] = None

    #: The default dtype to use for index data arrays. Hard coded to ``int32``,
    #: which should be sufficient for all datasets. ``int16`` is too small for
    #: many datasets, even after slicing.
    sensible_dtype = numpy.int32

    def __repr__(self) -> str:
        attrs = {
            'nodes': self.node_count,
            'edges': self.edge_count if self.has_edge_dimension else 'na',
            'faces': self.face_count,
        }
        attr_str = ' '.join(f'{key}: {value!r}' for key, value in attrs.items())
        return f'<{type(self).__name__} {attr_str}>'

    @cached_property
    def mesh_variable(self) -> xarray.DataArray:
        """
        Get the dummy variable that stores the mesh topology information.
        This is variable is either named using the ``topology_key`` argument to
        the contructor, or found by looking for a variable with the attribute
        ``cf_role`` of ``"mesh_topology"``.
        """
        if self.topology_key is not None:
            return self.dataset.data_vars[self.topology_key]
        try:
            return next(
                data_array for data_array in self.dataset.data_vars.values()
                if data_array.attrs.get('cf_role') == 'mesh_topology'
            )
        except StopIteration:
            raise ValueError("No mesh variable found")

    @property
    def mesh_attributes(self) -> Dict[Hashable, str]:
        """
        Get the mesh topology attributes from the dummy variable with the
        attribute ``cf_role`` of ``"mesh_topology"``.
        """
        return {
            key: value for key, value in self.mesh_variable.attrs.items()
            if isinstance(value, str)
        }

    @cached_property
    def sensible_fill_value(self) -> int:
        """
        A sensible value to use for ``_FillValue``.
        This is all-nines number larger than each of
        :attr:`node_count`, :attr:`edge_count`, and :attr:`face_count`.

        An alternate, simpler implementation would use ``numpy.iinfo(dtype).max``
        (the maximum integer value for a given dtype),
        but using all-nines is traditional.
        """
        # There will always be more edges than faces, as one face is composed
        # of at least three edges. There are probably more edges than nodes.
        # Not all datasets define how many edges they contain, and computing
        # that can be expensive. The upper bound for the number of edges in the
        # dataset is face_count * max_node_count, which is used as an estimate.
        # This is the highest possible index used in the dataset
        max_count = max([self.node_count, self.face_count * self.max_node_count])

        # This way is more mathematically fun, but less performant
        # fill_value =  10 ** (math.floor(math.log10(max_count)) + 2) - 1
        return int('9' * (len(str(max_count)) + 1))

    @cached_property
    def _node_coordinates(self) -> Tuple[Hashable, Hashable]:
        return _split_coord(self.mesh_attributes['node_coordinates'])

    @cached_property
    def _edge_coordinates(self) -> Tuple[Hashable, Hashable]:
        return _split_coord(self.mesh_attributes['edge_coordinates'])

    @cached_property
    def _face_coordinates(self) -> Tuple[Hashable, Hashable]:
        return _split_coord(self.mesh_attributes['face_coordinates'])

    @property
    def node_x(self) -> xarray.DataArray:
        """Data array of node X / longitude coordinates."""
        return self.dataset.data_vars[self._node_coordinates[0]]

    @property
    def node_y(self) -> xarray.DataArray:
        """Data array of node Y / latitude coordinates."""
        return self.dataset.data_vars[self._node_coordinates[1]]

    @property
    def edge_x(self) -> Optional[xarray.DataArray]:
        """Data array of characteristic edge X / longitude coordinates. Optional."""
        try:
            return self.dataset.data_vars[self._edge_coordinates[0]]
        except KeyError:
            return None

    @property
    def edge_y(self) -> Optional[xarray.DataArray]:
        """Data array of characteristic edge y / latitude coordinates. Optional."""
        try:
            return self.dataset.data_vars[self._edge_coordinates[1]]
        except KeyError:
            return None

    @property
    def face_x(self) -> Optional[xarray.DataArray]:
        """Data array of characteristic face x / longitude coordinates. Optional."""
        try:
            return self.dataset.data_vars[self._face_coordinates[0]]
        except KeyError:
            return None

    @property
    def face_y(self) -> Optional[xarray.DataArray]:
        """Data array of characteristic face y / latitude coordinates. Optional."""
        try:
            return self.dataset.data_vars[self._face_coordinates[1]]
        except KeyError:
            return None

    def _to_index_array(
        self,
        data_array: xarray.DataArray,
        primary_dimension: Hashable,
    ) -> numpy.ndarray:
        """
        Convert a data array of node, edge, or face indices
        into a masked numpy integer array.
        This takes care of casting arrays to integers
        and fixing one-based indexing where appropriate.
        """
        # Ensure the dimensions are in the correct order first
        if primary_dimension not in data_array.dims:
            raise ConventionViolationError(
                f"Data array {data_array.name!r} did not contain "
                f"primary dimension {primary_dimension!r}")
        if data_array.dims[0] != primary_dimension:
            data_array = data_array.transpose()

        values = data_array.values

        if not issubclass(values.dtype.type, numpy.integer):
            # If a data array has a fill value, xarray will convert that data array
            # to a floating point data type, and replace masked values with numpy.nan.
            # Here we convert a floating point array to a masked integer array.
            masked_values = numpy.ma.masked_invalid(values)
            # numpy will emit a warning when converting an array with numpy.nan to int,
            # even if the nans are masked out.
            masked_values.data[masked_values.mask] = self.sensible_fill_value
            values = masked_values.astype(self.sensible_dtype)
        elif '_FillValue' in data_array.attrs:
            # The data array has a fill value, but xarray has not applied it.
            # This implied the dataset was opened with mask_and_scale=False,
            # or was constructed in memory.
            # Convert it to a mask array with _FillValue masked out.
            values = numpy.ma.masked_equal(values, data_array.attrs['_FillValue'])
        else:
            # If the value is still an integer then it had no fill value.
            # Convert it to a mask array with no masked values.
            values = numpy.ma.masked_array(values, mask=numpy.ma.nomask)

        # UGRID conventions allow for zero based or one based indexing.
        # To be consistent we convert all indices to zero based.
        start_index = _get_start_index(data_array)
        if start_index != 0:
            values = values - start_index

        return values

    @cached_property
    def has_valid_edge_node_connectivity(self) -> bool:
        """
        Does the dataset contain a valid edge_node_connectivity variable?
        Some datasets have an attribute naming a edge_node_connectivity
        variable, but the variable doesn't exist. Other times, the shape of the
        variable is incorrect.
        """
        if not self.has_edge_dimension:
            return False

        try:
            data_array = self.dataset.data_vars[self.mesh_attributes["edge_node_connectivity"]]
        except KeyError:
            return False

        actual = set(data_array.dims)
        expected = {self.edge_dimension, self.two_dimension}
        if actual != expected:
            warnings.warn(
                f"Got a edge_node_connectivity variable {data_array.name!r} with "
                f"unexpected dimensions {actual}, expecting {expected}",
                ConventionViolationWarning,
            )
            return False

        return True

    @cached_property
    def edge_node_connectivity(self) -> xarray.DataArray:
        """
        This data array defines unique indexes for each edge. This allows data
        to be stored 'on' an edge.

        If this variable is *not* defined,
        data are not stored on the edges for this dataset.
        """
        if not self.has_edge_dimension:
            raise NoEdgeDimensionException

        if not self.has_valid_edge_node_connectivity:
            raise NoConnectivityVariableException(
                "No valid edge_node_connectivity defined")

        name = self.mesh_attributes['edge_node_connectivity']
        return self.dataset.data_vars[name]

    @cached_property
    def edge_node_array(self) -> numpy.ndarray:
        """
        An integer numpy array with shape
        (:attr:`edge_dimension`, 2),
        indicating which nodes each edge consists of.
        If data are stored on edges in this dataset,
        the ordering of these edges matches the ordering of the data.

        If this variable is *not* defined,
        we can derive it from the existing data,
        arbitrarily assigning an index to each edge.
        This will allow us to derive other connectivity variables if required.
        """
        if not self.has_edge_dimension:
            raise NoEdgeDimensionException

        if self.has_valid_edge_node_connectivity:
            return self._to_index_array(
                self.edge_node_connectivity, self.edge_dimension)

        return self.make_edge_node_array()

    @utils.timed_func
    def make_edge_node_array(self) -> numpy.ndarray:
        # Each edge is composed of two nodes. Each edge may be named twice,
        # once for each face. To de-duplicate this, edges are built up using
        # this dict-of-sets, where the dict index is the node with the
        # lower index, and the set is the node indices of the other end.
        low_highs: Dict[int, Set[int]] = defaultdict(set)

        for face_index, node_pairs in self._face_and_node_pair_iter():
            for pair in node_pairs:
                low, high = sorted(pair)
                low_highs[low].add(high)
        return numpy.array([
            [low, high]
            for low, highs in low_highs.items()
            for high in highs
        ], dtype=self.sensible_dtype)

    @cached_property
    def has_valid_edge_face_connectivity(self) -> bool:
        """
        Does the dataset contain a valid edge_face_connectivity variable?
        Some datasets have an attribute naming a edge_face_connectivity
        variable, but the variable doesn't exist. Other times, the shape of the
        variable is incorrect.
        """
        if not self.has_edge_dimension:
            return False

        try:
            data_array = self.dataset.data_vars[self.mesh_attributes["edge_face_connectivity"]]
        except KeyError:
            return False

        actual = set(data_array.dims)
        expected = {self.edge_dimension, self.two_dimension}
        if actual != expected:
            warnings.warn(
                f"Got a face_face_connectivity variable {data_array.name!r} with "
                f"unexpected dimensions {actual}, expecting {expected}",
                ConventionViolationWarning,
            )
            return False

        return True

    @cached_property
    def edge_face_connectivity(self) -> xarray.DataArray:
        """
        This data array shows which faces an edge borders on.
        """
        if not self.has_valid_edge_face_connectivity:
            raise NoConnectivityVariableException(
                "No valid edge_face_connectivity defined")
        name = self.mesh_attributes['edge_face_connectivity']
        return self.dataset.data_vars[name]

    @cached_property
    def edge_face_array(self) -> numpy.ndarray:
        """
        An integer numpy array with shape
        (:attr:`edge_dimension`, 2),
        indicating which faces border each edge.
        """
        if self.has_valid_edge_face_connectivity:
            return self._to_index_array(
                self.edge_face_connectivity, self.edge_dimension)

        return self.make_edge_face_array()

    @utils.timed_func
    def make_edge_face_array(self) -> numpy.ndarray:
        # The edge_face connectivity matrix
        shape = (self.edge_count, 2)
        filled = numpy.full(shape, self.sensible_fill_value, dtype=self.sensible_dtype)
        edge_face: numpy.ndarray = numpy.ma.masked_array(filled, mask=True)

        # The number of faces already seen for this edge
        edge_face_count = numpy.zeros(self.edge_count, dtype=self.sensible_dtype)

        for face_index, edge_indices in enumerate(self.face_edge_array):
            for edge_index in edge_indices.compressed():
                edge_face[edge_index, edge_face_count[edge_index]] = face_index
                edge_face_count[edge_index] += 1

        return edge_face

    @cached_property
    def has_valid_face_node_connectivity(self) -> bool:
        """
        Does the dataset contain a valid face_node_connectivity variable?
        If this is invalid, the entire dataset is invalid.
        """
        try:
            data_array = self.dataset.data_vars[self.mesh_attributes["face_node_connectivity"]]
        except KeyError:
            return False

        actual = set(data_array.dims)
        expected = {self.face_dimension, self.max_node_dimension}
        if actual != expected:
            warnings.warn(
                f"Got a face_node_connectivity variable {data_array.name!r} with "
                f"unexpected dimensions {actual}, expecting {expected}",
                ConventionViolationWarning,
            )
            return False

        return True

    @cached_property
    def face_node_connectivity(self) -> xarray.DataArray:
        """
        A variable that lists the nodes that make up the boundary of each face.
        This is the only required data variable in a UGRID dataset,
        all the others can be derived from this if required.
        """
        name = self.mesh_attributes['face_node_connectivity']
        return self.dataset.data_vars[name]

    @cached_property
    def face_node_array(self) -> numpy.ndarray:
        """
        An integer numpy array with shape
        (:attr:`face_dimension`, :attr:`max_node_dimension`),
        representing the node indices that make up each face.
        """
        return self._to_index_array(
            self.face_node_connectivity, self.face_dimension)

    @cached_property
    def has_valid_face_edge_connectivity(self) -> bool:
        """
        Does the dataset contain a valid face_edge_connectivity variable?
        Some datasets have an attribute naming a face_edge_connectivity
        variable, but the variable doesn't exist. Other times, the shape of the
        variable is incorrect.
        """
        try:
            data_array = self.dataset.data_vars[self.mesh_attributes["face_edge_connectivity"]]
        except KeyError:
            return False

        actual = set(data_array.dims)
        expected = {self.face_dimension, self.max_node_dimension}
        if actual != expected:
            warnings.warn(
                f"Got a face_edge_connectivity variable {data_array.name!r} with "
                f"unexpected dimensions {actual}, expecting {expected}",
                ConventionViolationWarning,
            )
            return False

        return True

    @cached_property
    def face_edge_connectivity(self) -> xarray.DataArray:
        """
        The face_edge_connectivity variable from the dataset, if present.
        """
        if not self.has_valid_face_edge_connectivity:
            raise NoConnectivityVariableException(
                "No valid face_edge_connectivity defined")
        name = self.mesh_attributes['face_edge_connectivity']
        return self.dataset.data_vars[name]

    @cached_property
    def face_edge_array(self) -> numpy.ndarray:
        """
        An integer numpy array with shape
        (:attr:`face_dimension`, :attr:`max_node_dimension`),
        representing the edge indices that border each face.
        """
        if self.has_valid_face_edge_connectivity:
            return self._to_index_array(
                self.face_edge_connectivity, self.face_dimension)

        return self.make_face_edge_array()

    @utils.timed_func
    def make_face_edge_array(self) -> numpy.ndarray:
        # Build a face_edge_connectivity matrix
        shape = (self.face_count, self.max_node_count)
        filled = numpy.full(shape, self.sensible_fill_value, dtype=self.sensible_dtype)
        face_edge: numpy.ndarray = numpy.ma.masked_array(filled, mask=True)

        node_pair_to_edge_index = {
            frozenset(edge): edge_index
            for edge_index, edge in enumerate(self.edge_node_array)
        }

        for face_index, node_pairs in self._face_and_node_pair_iter():
            for column, node_pair in enumerate(node_pairs):
                edge_index = node_pair_to_edge_index[frozenset(node_pair)]
                face_edge[face_index, column] = edge_index

        return face_edge

    @cached_property
    def has_valid_face_face_connectivity(self) -> bool:
        """
        Does the dataset contain a valid face_face_connectivity variable?
        Some datasets have an attribute naming a face_face_connectivity
        variable, but the variable doesn't exist. Other times, the shape of the
        variable is incorrect.
        """
        try:
            data_array = self.dataset.data_vars[self.mesh_attributes["face_face_connectivity"]]
        except KeyError:
            return False

        actual = set(data_array.dims)
        expected = {self.face_dimension, self.max_node_dimension}
        if actual != expected:
            warnings.warn(
                f"Got a face_face_connectivity variable {data_array.name!r} with "
                f"unexpected dimensions {actual}, expecting {expected}",
                ConventionViolationWarning,
            )
            return False

        return True

    @cached_property
    def face_face_connectivity(self) -> xarray.DataArray:
        """
        The face_face_connectivity variable from the dataset, if present.
        """
        if not self.has_valid_face_face_connectivity:
            raise NoConnectivityVariableException(
                "No valid face_face_connectivity defined")
        name = self.mesh_attributes['face_face_connectivity']
        return self.dataset.data_vars[name]

    @cached_property
    def face_face_array(self) -> numpy.ndarray:
        """
        An integer numpy array with shape
        (:attr:`face_dimension`, :attr:`max_node_dimension`),
        representing the indices of all faces that border a given face.
        """
        if self.has_valid_face_face_connectivity:
            return self._to_index_array(
                self.face_face_connectivity, self.face_dimension)

        return self.make_face_face_array()

    def make_face_face_array(self) -> numpy.ndarray:
        # Build a face_face_connectivity matrix
        face_count = numpy.zeros(self.face_count, dtype=self.sensible_dtype)
        shape = (self.face_count, self.max_node_count)
        filled = numpy.full(shape, self.sensible_fill_value, dtype=self.sensible_dtype)
        face_face: numpy.ndarray = numpy.ma.masked_array(filled, mask=True)

        for edge_index, face_indices in enumerate(self.edge_face_array):
            if numpy.any(numpy.ma.getmask(face_indices)):
                continue
            left, right = face_indices
            face_face[left, face_count[left]] = right
            face_face[right, face_count[right]] = left
            face_count[left] += 1
            face_count[right] += 1

        return face_face

    def _face_and_node_pair_iter(self) -> Iterable[Tuple[int, List[Tuple[int, int]]]]:
        """
        An iterator returning a tuple of ``(face_index, edges)``,
        where ``edges`` is a list of ``(node_index, node_index)`` tuples
        defining the edges of the face.
        """
        face_node = self.face_node_array
        for face_index, node_indices in enumerate(face_node):
            node_indices = node_indices.compressed()
            node_indices = numpy.append(node_indices, node_indices[0])
            yield face_index, list(utils.pairwise(node_indices))

    @cached_property
    def dimension_for_grid_kind(self) -> Dict[UGridKind, Hashable]:
        """
        Get the dimension names for each of the grid types in this dataset.
        """
        dimensions = {
            UGridKind.face: self.face_dimension,
            UGridKind.node: self.node_dimension,
        }
        if self.has_edge_dimension:
            dimensions[UGridKind.edge] = self.edge_dimension
        return dimensions

    @cached_property
    def two_dimension(self) -> Hashable:
        """
        Get the name of the dimension with size two, for things like edge
        connectivity. The standard name for this dimension is 'Two'.
        """
        two = 'Two'
        # Check for the standard name
        if two in self.dataset.dims and self.dataset.dims[two] == 2:
            return two
        # Check for any other dimension of size 2
        for name, size in self.dataset.dims.items():
            if size == 2:
                return name
        # Make up a new dimension with the standard name
        return two

    @property
    def node_dimension(self) -> Hashable:
        """The name of the dimension for the number of nodes."""
        # This is implicitly given by some of the required variables, such as
        # the variable for node x-coordinates.
        return self.node_x.dims[0]

    @property
    def has_edge_dimension(self) -> bool:
        """
        Check whether this dataset has an edge dimension.
        If not, data are not stored on edges in this dataset.
        """
        if 'edge_dimension' in self.mesh_attributes:
            return True

        # We do not use `self.has_valid_edge_node_connectivity` here to prevent
        # circular logic. This leaves open the possibility of a dataset having
        # an invalid edge_node_connectivity variable without a proper edge
        # dimension defined, but this attribute being True regardless.
        topo_keys = ['edge_node_connectivity', 'edge_face_connectivity']
        return any(
            key in self.mesh_attributes
            and self.mesh_attributes[key] in self.dataset.variables
            for key in topo_keys
        )

    @cached_property
    def edge_dimension(self) -> Hashable:
        """
        The name of the dimension for the number of edges.
        """
        if not self.has_edge_dimension:
            raise NoEdgeDimensionException

        with suppress(KeyError):
            return self.mesh_attributes['edge_dimension']

        # If there is no edge_dimension attribute,
        # the edge dimension is implicitly defined as the first dimension
        # on either of these variables.
        # Both of these variables are optional,
        # leaving the possibility that there is no edge dimension defined.
        topo_keys = ['edge_node_connectivity', 'edge_face_connectivity']
        names = (self.mesh_attributes[key] for key in topo_keys if key in self.mesh_attributes)
        variables = (self.dataset.variables[name] for name in names if name in self.dataset.variables)

        try:
            return next(variable.dims[0] for variable in variables)
        except StopIteration:
            # This should have already happened above, but just in case.
            raise NoEdgeDimensionException

    @cached_property
    def face_dimension(self) -> Hashable:
        """The name of the dimension for the number of faces."""
        try:
            # By definition this is either the dimension named in this attribute ...
            return self.mesh_attributes['face_dimension']
        except KeyError:
            # ... Or the first dimension in this required variable
            return self.face_node_connectivity.dims[0]

    @property
    def max_node_dimension(self) -> Hashable:
        """The name of the dimension for the maximum nodes / edges per face."""
        # This dimension is not named as part of the UGRID spec, but it is
        # always the other dimension in the `face_node_connectivity` variable.
        dims = set(self.face_node_connectivity.dims)
        assert len(dims) == 2
        dims.remove(self.face_dimension)
        return dims.pop()

    @property
    def node_count(self) -> int:
        """The number of nodes in the dataset."""
        return self.dataset.dims[self.node_dimension]

    @property
    def edge_count(self) -> int:
        """The number of edges in the dataset."""
        if not self.has_edge_dimension:
            raise NoEdgeDimensionException()

        # This dimension may not be defined, so ignore KeyErrors. We can
        # compute it below.
        with suppress(KeyError):
            return self.dataset.dims[self.edge_dimension]

        # By computing the edge_node array we can determine how many edges exist
        return self.edge_node_array.shape[0]

    @property
    def face_count(self) -> int:
        """
        The number of faces in the dataset.
        """
        return self.dataset.dims[self.face_dimension]

    @property
    def max_node_count(self) -> int:
        """The maximum number of nodes / edges per face."""
        return self.dataset.dims[self.max_node_dimension]


class UGridKind(str, enum.Enum):
    """UGRID datasets can store data on faces, edges, or nodes."""
    face = 'face'
    edge = 'edge'
    node = 'node'


#: UGRID indices are always single integers, for all index kinds.
UGridIndex = Tuple[UGridKind, int]


class UGrid(DimensionConvention[UGridKind, UGridIndex]):
    """A :class:`.Convention` subclass to handle unstructured grid datasets.
    """

    default_grid_kind = UGridKind.face

    @classmethod
    def check_dataset(cls, dataset: xarray.Dataset) -> Optional[int]:
        """
        A UGrid dataset needs a global attribute of Conventions = 'UGRID/...',
        and a variable with attribute cf_role = 'mesh_topology'
        and topology_dimension = 2
        """
        conventions = str(dataset.attrs.get('Conventions', ''))
        if 'UGRID' not in conventions:
            return None

        topology = Mesh2DTopology(dataset)
        try:
            mesh = topology.mesh_variable
        except ValueError:
            return None

        if mesh.attrs.get('topology_dimension') != 2:
            return None

        return Specificity.HIGH

    @cached_property
    def topology(self) -> Mesh2DTopology:
        """
        The :class:`Mesh2DTopology` of this dataset - which faces connect with which edges.
        """
        return Mesh2DTopology(self.dataset)

    @cached_property
    def grid_dimensions(self) -> Dict[UGridKind, Sequence[Hashable]]:
        dimensions: Dict[UGridKind, Sequence[Hashable]] = {
            UGridKind.node: [self.topology.node_dimension],
            UGridKind.face: [self.topology.face_dimension],
        }
        if self.topology.has_edge_dimension:
            dimensions[UGridKind.edge] = [self.topology.edge_dimension]
        return dimensions

    def unpack_index(self, index: UGridIndex) -> Tuple[UGridKind, Sequence[int]]:
        return index[0], index[1:]

    def pack_index(self, grid_kind: UGridKind, indices: Sequence[int]) -> UGridIndex:
        return (grid_kind, indices[0])

    @cached_property
    def grid_kinds(self) -> FrozenSet[UGridKind]:
        items = [UGridKind.face, UGridKind.node]
        # The edge dimension is optional, not all UGRID datasets define it
        if self.topology.has_edge_dimension:
            items.append(UGridKind.edge)
        return frozenset(items)

    @cached_property
    @utils.timed_func
    def polygons(self) -> numpy.ndarray:
        """Generate list of Polygons"""
        # X,Y coords of each node
        topology = self.topology
        node_x = topology.node_x.values
        node_y = topology.node_y.values
        face_node = topology.face_node_array
        polygons = numpy.full(topology.face_count, None, dtype=numpy.object_)

        # `shapely.polygons` will make polygons with the same number of vertices.
        # UGRID polygons have arbitrary numbers of vertices.
        # Group polygons by how many vertices they have, then make them in bulk.
        polygons_of_size: Mapping[int, Dict[int, numpy.ndarray]] = defaultdict(dict)
        for index, row in enumerate(face_node):
            vertices = row.compressed()
            polygons_of_size[vertices.size][index] = numpy.c_[node_x[vertices], node_y[vertices]]

        for size, size_polygons in polygons_of_size.items():
            coords = numpy.stack(list(size_polygons.values()))
            shapely.polygons(coords, indices=list(size_polygons.keys()), out=polygons)

        return polygons

    @cached_property
    def face_centres(self) -> numpy.ndarray:
        face_x, face_y = self.topology.face_x, self.topology.face_y
        if face_x is not None and face_y is not None:
            face_centres = numpy.column_stack((face_x, face_y))
            return cast(numpy.ndarray, face_centres)
        return super().face_centres

    @cached_property
    def bounds(self) -> Bounds:
        topology = self.topology
        min_x = numpy.nanmin(topology.node_x)
        max_x = numpy.nanmax(topology.node_x)
        min_y = numpy.nanmin(topology.node_y)
        max_y = numpy.nanmax(topology.node_y)
        return (min_x, min_y, max_x, max_y)

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xarray.Dataset:
        """
        Make a mask dataset from a clip geometry for this dataset.
        This mask can later be applied using :meth:`apply_clip_mask`.

        A mask for a UGRID dataset indicates which nodes, edges, and faces
        (collectively 'elements') to include, and their new indices. When
        elements are dropped from the dataset, to keep the remaining data
        contiguous, all elements gain new indices.

        The mask consists of three data arrays corresponding to nodes, edges,
        and faces. Each data array indicates the new index for that element, or
        whether that element should be dropped.

        Making a mask consists of three steps:

        * Identifying which polygons to include
        * Adding a buffer zone around these faces
        * Making a mask for the faces, edges, and nodes
        """
        # Find all faces that intersect the clip geometry
        logger.info("Making clip mask")
        face_indices = self.strtree.query(clip_geometry, predicate='intersects')
        logger.debug("Found %d intersecting faces, adding size %d buffer...", len(face_indices), buffer)

        # Include all the neighbours of the intersecting faces
        for _ in range(buffer):
            face_indices = buffer_faces(face_indices, self.topology)
        logger.debug("Total faces in mask: %d", len(face_indices))

        # Make a mask dataset
        return mask_from_face_indices(face_indices, self.topology)

    def apply_clip_mask(self, clip_mask: xarray.Dataset, work_dir: Pathish) -> xarray.Dataset:
        """
        Make a new dataset by applying a clip mask to this dataset.

        See Also
        --------
        :meth:`make_clip_mask`
        :func:`mask_from_face_indices`
        """
        logger.info("Applying clip mask")
        dataset = self.dataset
        topology = self.topology
        work_path = pathlib.Path(work_dir)

        # Each variable will be sliced and saved to a separate file in the
        # working directory, then reassembled later as a multifile dataset.
        # This list keeps track of all the individual files.
        mfdataset_paths = []

        logger.debug("Slicing topology variables...")
        # Collect all the topology variables here. These need special handling,
        # compared to data variables. The mesh variable can be reused without
        # any changes.
        topology_variables: List[xarray.DataArray] = [topology.mesh_variable]

        # This is the fill value used in the mask.
        new_fill_value = clip_mask.data_vars['new_node_index'].encoding['_FillValue']

        def integer_indices(data_array: xarray.DataArray) -> numpy.ndarray:
            masked_values = numpy.ma.masked_invalid(data_array.values)
            # numpy will emit a warning when converting an array with numpy.nan to int,
            # even if the nans are masked out.
            masked_values.data[masked_values.mask] = new_fill_value
            masked_integers: numpy.ndarray = masked_values.astype(numpy.int_)
            return masked_integers

        new_node_indices = integer_indices(clip_mask.data_vars['new_node_index'])
        new_face_indices = integer_indices(clip_mask.data_vars['new_face_index'])
        has_edges = 'new_edge_index' in clip_mask.data_vars
        if has_edges:
            new_edge_indices = integer_indices(clip_mask.data_vars['new_edge_index'])

        # Re-index the face_node_connectivity variable
        topology_variables.append(update_connectivity(
            topology.face_node_connectivity, topology.face_node_array,
            new_face_indices, new_node_indices,
            primary_dimension=topology.face_dimension, fill_value=new_fill_value))

        # Re-index each of the optional connectivity variables
        if has_edges and topology.has_valid_face_edge_connectivity:
            topology_variables.append(update_connectivity(
                topology.face_edge_connectivity, topology.face_edge_array,
                new_face_indices, new_edge_indices,
                primary_dimension=topology.edge_dimension, fill_value=new_fill_value))

        if topology.has_valid_face_face_connectivity:
            topology_variables.append(update_connectivity(
                topology.face_face_connectivity, topology.face_face_array,
                new_face_indices, new_face_indices,
                primary_dimension=topology.face_dimension, fill_value=new_fill_value))

        if has_edges and topology.has_valid_edge_face_connectivity:
            topology_variables.append(update_connectivity(
                topology.edge_face_connectivity, topology.edge_face_array,
                new_edge_indices, new_face_indices,
                primary_dimension=topology.face_dimension, fill_value=new_fill_value))

        if has_edges and topology.has_valid_edge_node_connectivity:
            topology_variables.append(update_connectivity(
                topology.edge_node_connectivity, topology.edge_node_array,
                new_edge_indices, new_node_indices,
                primary_dimension=topology.edge_dimension, fill_value=new_fill_value))

        # Save all the topology variables to one combined dataset
        topology_path = work_path / (str(topology.mesh_variable.name) + ".nc")
        topology_dataset = xarray.Dataset(
            data_vars={variable.name: variable for variable in topology_variables},
            coords=dataset.coords,
        )
        topology_dataset.to_netcdf(topology_path)
        mfdataset_paths.append(topology_path)

        topology_variable_names = list(topology_dataset.variables.keys())
        del topology_dataset
        del topology_variables

        logger.debug("Slicing data variables...")
        dimension_masks: Dict[Hashable, numpy.ndarray] = {
            topology.node_dimension: ~numpy.ma.getmask(new_node_indices),
            topology.face_dimension: ~numpy.ma.getmask(new_face_indices),
        }
        if has_edges:
            dimension_masks[topology.edge_dimension] = ~numpy.ma.getmask(new_edge_indices)
        mesh_dimensions = set(dimension_masks.keys())

        for name, data_array in dataset.data_vars.items():
            data_array_path = work_path / (str(name) + '.nc')
            if name in topology_variable_names:
                logger.debug("Skipping %r as it is a topology variable", name)

            elif set(data_array.dims).isdisjoint(mesh_dimensions):
                logger.debug("Using %r as-is, as it has no mesh dimensions", name)
                data_array = data_array.copy()
                utils.disable_default_fill_value(data_array)
                data_array.to_netcdf(data_array_path)
                mfdataset_paths.append(data_array_path)
                del data_array

            else:
                logger.debug("Slicing %r, dimensions %r", name, data_array.dims)
                # This is basically doing `data_array.isel({dimension: rows})`,
                # except that `isel` does this _very_ slowly.
                values = data_array.values
                for index, dim in enumerate(data_array.dims):
                    # For each dimension in the data array, if it is one of
                    # the dimensions that we want to subset on...
                    if dim in dimension_masks:
                        # ... make a tuple slice like (:, :, :, bool_array)
                        # where bool_array indicates which rows to include
                        slice_index = tuple([numpy.s_[:]] * index + [dimension_masks[dim]])  # type: ignore
                        values = values[slice_index]

                data_array = xarray.DataArray(data=values, dims=data_array.dims, name=name)
                utils.disable_default_fill_value(data_array)
                data_array.to_netcdf(data_array_path)
                mfdataset_paths.append(data_array_path)

                del data_array
                del values

        logger.debug("Merging individual variables...")
        new_dataset = xarray.open_mfdataset(mfdataset_paths, lock=False)
        return utils.dataset_like(dataset, new_dataset)

    def get_all_geometry_names(self) -> List[Hashable]:
        topology = self.topology

        names = [
            topology.mesh_variable.name,
            topology.face_node_connectivity.name,
            topology.node_x.name,
            topology.node_y.name,
        ]
        if topology.has_valid_face_edge_connectivity:
            names.append(topology.face_edge_connectivity.name)
        if topology.has_valid_face_face_connectivity:
            names.append(topology.face_face_connectivity.name)
        if topology.has_valid_edge_node_connectivity:
            names.append(topology.edge_node_connectivity.name)
        if topology.has_valid_edge_face_connectivity:
            names.append(topology.edge_face_connectivity.name)
        if topology.edge_x is not None:
            names.append(topology.edge_x.name)
        if topology.edge_y is not None:
            names.append(topology.edge_y.name)
        if topology.face_x is not None:
            names.append(topology.face_x.name)
        if topology.face_y is not None:
            names.append(topology.face_y.name)
        return names

    def drop_geometry(self) -> xarray.Dataset:
        dataset = super().drop_geometry()
        dataset.attrs.pop('Conventions', None)
        return dataset
