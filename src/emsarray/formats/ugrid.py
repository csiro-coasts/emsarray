"""
Unstructured grid format.

See also
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
    Any, Dict, FrozenSet, Hashable, Iterable, List, Optional, Set, Tuple, cast
)

import numpy as np
import xarray as xr
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from emsarray import utils
from emsarray.types import Pathish

from ._base import Format
from ._helpers import Specificity, register_format

logger = logging.getLogger(__name__)


def _split_coord(attr: str) -> Tuple[str, str]:
    x, y = attr.split(None, 1)
    return (x, y)


def buffer_faces(face_indices: np.ndarray, topology: Mesh2DTopology) -> np.ndarray:
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
    face_node = topology.face_node_connectivity

    # Find all nodes for the faces
    included_nodes = set(np.unique(face_node.values[face_indices]))
    included_nodes.discard(face_node.attrs.get('_FillValue'))

    # Find all faces that are composed of any of the nodes just found
    included_faces = (
        face_index
        for face_index, node_indices in enumerate(face_node.values)
        # Either one of the original faces ...
        if face_index in original_face_indices
        # ... or shares a node with one of the original faces
        or bool(included_nodes.intersection(node_indices))
    )
    return cast(np.ndarray, np.fromiter(included_faces, dtype=topology.sensible_dtype))


def mask_from_face_indices(face_indices: np.ndarray, topology: Mesh2DTopology) -> xr.Dataset:
    """
    Make a mask dataset from a list of face indices.
    This mask can later be applied using :meth:`~.Format.apply_clip_mask`.

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

    def new_element_indices(size: int, indices: np.ndarray) -> np.ndarray:
        new_indices = np.full(
            (size,), fill_value=fill_value, dtype=topology.sensible_dtype)
        new_indices[indices] = np.arange(len(indices))
        return new_indices

    # Record which old face index maps to which new face index
    data_vars['new_face_index'] = xr.DataArray(
        data=new_element_indices(topology.face_count, face_indices),
        dims=['old_face_index'],
        attrs={'_FillValue': fill_value},
    )

    # Find all edges associated with included faces. Don't bother if the
    # dataset doesn't define an edge dimension
    if topology.has_edge_dimension:
        face_edge = topology.face_edge_connectivity
        face_edge_fill_value = face_edge.attrs.get('_FillValue')
        edges_set = set()
        for face_index in face_indices.tolist():
            edge_indices = face_edge.values[face_index]
            edges_set.update(edge_indices[edge_indices != face_edge_fill_value].tolist())
        edge_indices = np.array(sorted(edges_set))

        # Record which old edge index maps to which new edge index
        data_vars['new_edge_index'] = xr.DataArray(
            data=new_element_indices(topology.edge_count, edge_indices),
            dims=['old_edge_index'],
            attrs={'_FillValue': fill_value},
        )

    # Find all nodes associated with included faces
    face_node = topology.face_node_connectivity
    face_node_fill_value = face_node.attrs.get('_FillValue')
    nodes_set = set()
    for face_index in face_indices.tolist():
        node_indices = face_node.values[face_index]
        nodes_set.update(node_indices[node_indices != face_node_fill_value].tolist())
    node_indices = np.array(sorted(nodes_set))

    # Record which old node index maps to which new node index
    data_vars['new_node_index'] = xr.DataArray(
        data=new_element_indices(topology.node_count, node_indices),
        dims=['old_node_index'],
        attrs={'_FillValue': fill_value},
    )

    # Make the mask dataset
    return xr.Dataset(data_vars=data_vars, attrs={
        'title': 'UGRID dataset mask',
    })


class NoEdgeDimensionException(ValueError):
    """
    Raised when the dataset does not define a name for the edge dimension, and
    no dimension name can be derived from the available connectivity variables.
    """


@dataclass
class Mesh2DTopology:
    """
    A helper for extracting the topology data for a 2D unstructured mesh
    dataset. The dataset will contain a dummy variable with the attribute
    `cf_role = "mesh_topology"`. The attributes of that variable explain the
    topology of the dataset. For example, the names of the variables that store
    the coordinates for nodes can be found in the `node_coordinates` attribute
    as the string "x_variable y_variable".
    """
    #: The UGRID dataset
    dataset: xr.Dataset

    #: The name of the mesh topology variable. Optional. If not provided, the
    #: mesh topology dummy variable will be found by checking the `cf_role`
    #: attribute.
    topology_key: Optional[str] = None

    #: The default dtype to use for index data arrays. Hard coded to ``int32``,
    #: which should be sufficient for all datasets. ``int16`` is too small for
    #: many datasets, even after slicing.
    sensible_dtype = np.int32

    def __repr__(self) -> str:
        attrs = {
            'nodes': self.node_count,
            'edges': self.edge_count if self.has_edge_dimension else 'na',
            'faces': self.face_count,
        }
        attr_str = ' '.join(f'{key}: {value!r}' for key, value in attrs.items())
        return f'<{type(self).__name__} {attr_str}>'

    @cached_property
    def mesh_variable(self) -> xr.DataArray:
        """
        Get the dummy variable that stores the mesh topology information.
        This is variable is either named using the `topology_key` argument to
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
        attribute `cf_role` of "mesh_topology.
        """
        return {
            key: value for key, value in self.mesh_variable.attrs.items()
            if isinstance(value, str)
        }

    @cached_property
    def sensible_fill_value(self) -> int:
        """
        Find a sensible value to use for _FillValue. This finds an all-nines
        number larger than each of node_count, edge_count, and face_count.

        An alternate, simpler implementation would use `np.iinfo(dtype).max`
        (the maximum integer value for a given dtype), but using all-nines is
        traditional.
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
    def _node_coordinates(self) -> Tuple[str, str]:
        return _split_coord(cast(str, self.mesh_attributes['node_coordinates']))

    @cached_property
    def _edge_coordinates(self) -> Tuple[str, str]:
        return _split_coord(cast(str, self.mesh_attributes['edge_coordinates']))

    @cached_property
    def _face_coordinates(self) -> Tuple[str, str]:
        return _split_coord(cast(str, self.mesh_attributes['face_coordinates']))

    @property
    def node_x(self) -> xr.DataArray:
        """Data array of node X / longitude coordinates."""
        return self.dataset.data_vars[self._node_coordinates[0]]

    @property
    def node_y(self) -> xr.DataArray:
        """Data array of node Y / latitude coordinates."""
        return self.dataset.data_vars[self._node_coordinates[1]]

    @property
    def edge_x(self) -> Optional[xr.DataArray]:
        """Data array of characteristic edge X / longitude coordinates. Optional."""
        try:
            return self.dataset.data_vars[self._edge_coordinates[0]]
        except KeyError:
            return None

    @property
    def edge_y(self) -> Optional[xr.DataArray]:
        """Data array of characteristic edge y / latitude coordinates. Optional."""
        try:
            return self.dataset.data_vars[self._edge_coordinates[1]]
        except KeyError:
            return None

    @property
    def face_x(self) -> Optional[xr.DataArray]:
        """Data array of characteristic face x / longitude coordinates. Optional."""
        try:
            return self.dataset.data_vars[self._face_coordinates[0]]
        except KeyError:
            return None

    @property
    def face_y(self) -> Optional[xr.DataArray]:
        """Data array of characteristic face y / latitude coordinates. Optional."""
        try:
            return self.dataset.data_vars[self._face_coordinates[1]]
        except KeyError:
            return None

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
                f"Got a face_face_connectivity variable {data_array.name!r} with "
                f"unexpected dimensions {actual}, expecting {expected}"
            )
            return False

        return True

    @cached_property
    def edge_node_connectivity(self) -> xr.DataArray:
        """
        This data array defines unique indexes for each edge. This allows data
        to be stored 'on' an edge.

        If this dataset is _not_ defined, data are not stored on the edges for
        this dataset. We can derive it from the existing data, arbitrarily
        assigning an index to each edge. This will allow us to derive other
        datasets if required.
        """
        if self.has_valid_edge_node_connectivity:
            return self.dataset.data_vars[self.mesh_attributes['edge_node_connectivity']]

        # Access this here so it raises an error early
        edge_dimension = self.edge_dimension

        logger.info("Building edge_node_connectivity")
        with utils.PerfTimer() as timer:
            # Each edge is composed of two nodes. Each edge may be named twice,
            # once for each face. To de-duplicate this, edges are built up using
            # this dict-of-sets, where the dict index is the node with the
            # lower index, and the set is the node indices of the other end.
            low_highs: Dict[int, Set[int]] = defaultdict(set)

            for face_index, node_pairs in self._face_and_node_pair_iter():
                for pair in node_pairs:
                    low, high = sorted(pair)
                    low_highs[low].add(high)
            edge_node = np.array([
                [low, high]
                for low, highs in low_highs.items()
                for high in highs
            ], dtype=self.sensible_dtype)
        logger.debug("Built edge_node_connectivity in %f seconds", timer.elapsed)

        return xr.DataArray(
            data=edge_node,
            dims=[edge_dimension, self.two_dimension],
            attrs={
                "cf_role": "edge_node_connectivity",
            },
        )

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
                f"unexpected dimensions {actual}, expecting {expected}"
            )
            return False

        return True

    @cached_property
    def edge_face_connectivity(self) -> xr.DataArray:
        """
        This data array shows which faces an edge borders on.
        """
        if self.has_valid_edge_face_connectivity:
            return self.dataset.data_vars[self.mesh_attributes['edge_face_connectivity']]

        # Access these outside of the timer below
        fill_value = self.sensible_fill_value
        face_edge = self.face_edge_connectivity
        face_edge_fill_value = face_edge.attrs.get('_FillValue')

        # Build an edge_face_connectivity matrix
        logger.info("Building edge_face_connectivity")
        with utils.PerfTimer() as timer:
            edge_face_count = np.zeros(self.edge_count, dtype=self.sensible_dtype)
            edge_face = np.full(
                (self.edge_count, 2), fill_value=fill_value, dtype=self.sensible_dtype)

            with np.nditer(face_edge.values, flags=['multi_index']) as nditer:
                for edge_index in nditer:
                    if edge_index == face_edge_fill_value:
                        continue
                    face_index = nditer.multi_index[0]
                    edge_face[edge_index, edge_face_count[edge_index]] = face_index
                    edge_face_count[edge_index] += 1
        logger.debug("Built edge_face_connectivity in %f seconds", timer.elapsed)

        return xr.DataArray(
            data=edge_face,
            dims=[self.edge_dimension, self.two_dimension],
            attrs={
                '_FillValue': fill_value,
                'cf_role': 'edge_face_connectivity',
            },
        )

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
                f"Got a face_face_connectivity variable {data_array.name!r} with "
                f"unexpected dimensions {actual}, expecting {expected}"
            )
            return False

        return True

    @cached_property
    def face_node_connectivity(self) -> xr.DataArray:
        """
        A dataset that lists the nodes that make up the boundary of each face.
        This is the only required data variable in a UGRID dataset, all the
        others can be derived from this if required.
        """
        return self.dataset.data_vars[self.mesh_attributes['face_node_connectivity']]

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
                f"Got a face_face_connectivity variable {data_array.name!r} with "
                f"unexpected dimensions {actual}, expecting {expected}"
            )
            return False

        return True

    @cached_property
    def face_edge_connectivity(self) -> xr.DataArray:
        try:
            return self.dataset.data_vars[self.mesh_attributes['face_edge_connectivity']]
        except KeyError:
            pass

        # Access these outside of the timer below
        fill_value = self.sensible_fill_value
        edge_node_connectivity = self.edge_node_connectivity

        # Build a face_edge_connectivity matrix
        logger.info("Building face_edge_connectivity")
        with utils.PerfTimer() as timer:
            edge_face = np.full(
                (self.face_count, self.max_node_count),
                fill_value=fill_value, dtype=self.sensible_dtype)
            node_pair_to_edge_index = {
                frozenset(edge): edge_index
                for edge_index, edge in enumerate(edge_node_connectivity.values)
            }

            for face_index, node_pairs in self._face_and_node_pair_iter():
                for column, node_pair in enumerate(node_pairs):
                    edge_index = node_pair_to_edge_index[frozenset(node_pair)]
                    edge_face[face_index, column] = edge_index
        logger.debug("Built face_edge_connectivity in %f seconds", timer.elapsed)

        return xr.DataArray(
            data=edge_face,
            dims=[self.face_dimension, self.max_node_dimension],
            attrs={
                '_FillValue': fill_value,
                'cf_role': 'face_edge_connectivity',
            }
        )

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
                f"unexpected dimensions {actual}, expecting {expected}"
            )
            return False

        return True

    @cached_property
    def face_face_connectivity(self) -> xr.DataArray:
        if self.has_valid_face_face_connectivity:
            return self.dataset.data_vars[self.mesh_attributes['face_face_connectivity']]

        # Access these outside of the timer below
        fill_value = self.sensible_fill_value
        edge_face_connectivity = self.edge_face_connectivity
        edge_face_fill_value = edge_face_connectivity.attrs.get('_FillValue')

        # Build a face_face_connectivity matrix
        logger.info("Building face_face_connectivity")
        with utils.PerfTimer() as timer:
            face_count = np.zeros(self.face_count, dtype=self.sensible_dtype)
            face_face = np.full(
                (self.face_count, self.max_node_count),
                fill_value=fill_value, dtype=self.sensible_dtype)

            for edge_index, face_indices in enumerate(edge_face_connectivity.values):
                if np.any(face_indices == edge_face_fill_value):
                    continue
                left, right = face_indices
                face_face[left, face_count[left]] = right
                face_face[right, face_count[right]] = left
                face_count[left] += 1
                face_count[right] += 1
        logger.debug("Built face_face_connectivity in %f seconds", timer.elapsed)

        return xr.DataArray(
            data=face_face,
            dims=[self.face_dimension, self.max_node_dimension],
            attrs={
                '_FillValue': fill_value,
                'cf_role': 'face_face_connectivity',
                'long_name': "Indicated which faces are neighbours",
                'start_index': 0,
            },
        )

    def _face_and_node_pair_iter(self) -> Iterable[Tuple[int, List[Tuple[int, int]]]]:
        """
        An iterator returning a tuple of ``(face_index, edges)``,
        where ``edges`` is a list of ``(node_index, node_index)`` tuples
        defining the edges of the face.
        """
        face_node = self.face_node_connectivity
        face_node_fill_value = face_node.attrs.get('_FillValue')
        for face_index, node_indices in enumerate(face_node.values):
            node_indices = node_indices[node_indices != face_node_fill_value]
            node_indices = np.append(node_indices, node_indices[0])
            yield face_index, list(utils.pairwise(node_indices))

    @cached_property
    def dimension_for_grid_kind(self) -> Dict[UGridKind, str]:
        dimensions = {
            UGridKind.face: self.face_dimension,
            UGridKind.node: self.node_dimension,
        }
        if self.has_edge_dimension:
            dimensions[UGridKind.edge] = cast(str, self.edge_dimension)
        return dimensions

    @cached_property
    def two_dimension(self) -> str:
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
                return str(name)
        # Make up a new dimension with the standard name
        return two

    @property
    def node_dimension(self) -> str:
        """The name of the dimension for the number of nodes."""
        # This is implicitly given by some of the required variables, such as
        # the variable for node x-coordinates.
        return str(self.node_x.dims[0])

    @property
    def has_edge_dimension(self) -> bool:
        if 'edge_dimension' in self.mesh_attributes:
            return True
        topo_keys = ['edge_node_connectivity', 'edge_face_connectivity']
        return any(
            key in self.mesh_attributes and self.mesh_attributes[key] in self.dataset.variables
            for key in topo_keys
        )

    @cached_property
    def edge_dimension(self) -> Optional[str]:
        """
        The name of the dimension for the number of edges.

        Returns None if there is no such dimension. This implies that data are
        not stored on edges in this dataset.
        """
        if not self.has_edge_dimension:
            raise NoEdgeDimensionException

        with suppress(KeyError):
            return self.mesh_attributes['edge_dimension']

        # If there is no edge dimension defined, it is implicitly defined as
        # the first dimension on either of these variables. Both of these
        # variables is optional, leaving the possibility that there is no edge
        # dimension defined.
        topo_keys = ['edge_node_connectivity', 'edge_face_connectivity']
        names = (self.mesh_attributes[key] for key in topo_keys if key in self.mesh_attributes)
        variables = (self.dataset.variables[name] for name in names if name in self.dataset.variables)

        try:
            return next(str(variable.dims[0]) for variable in variables)
        except StopIteration:
            # This should have already happened above, but just in case.
            raise NoEdgeDimensionException

    @cached_property
    def face_dimension(self) -> str:
        """The name of the dimension for the number of faces."""
        try:
            # By definition this is either the dimension named in this attribute ...
            return self.mesh_attributes['face_dimension']
        except KeyError:
            # ... Or the first dimension in this required variable
            return str(self.face_node_connectivity.dims[0])

    @property
    def max_node_dimension(self) -> str:
        """The name of the dimension for the maximum nodes / edges per face."""
        # This dimension is not named as part of the UGRID spec, but it is
        # always the other dimension in the `face_node_connectivity` variable.
        dims = set(self.face_node_connectivity.dims)
        assert len(dims) == 2
        dims.remove(self.face_dimension)
        return str(dims.pop())

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

        edge_node = self.edge_node_connectivity
        return edge_node.shape[edge_node.dims.index(self.edge_dimension)]

    @property
    def face_count(self) -> int:
        """The number of faces in the dataset."""
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


@register_format
class UGrid(Format[UGridKind, UGridIndex]):
    """A :class:`.Format` subclass to handle unstructured grid datasets.

    UGRID datasets must be opened with ``mask_and_scale=False``,
    as xarray does not handle masked integer variables well.
    :meth:`UGrid.open_dataset` or :func:`emsarray.open_dataset` can be useful here.
    """

    default_grid_kind = UGridKind.face

    @classmethod
    def open_dataset(cls, path: Pathish, **kwargs: Any) -> xr.Dataset:
        """
        Open the dataset at ``path``, setting ``mask_and_scale=False``.

        Example
        -------

        .. code-block:: python

            from emsarray.formats.ugrid import UGrid
            dataset = UGrid.open_dataset("./tests/datasets/ugrid_mesh2d.nc")

        See also
        --------
        :func:`emsarray.open_dataset`
        """
        return cast(xr.Dataset, xr.open_dataset(path, mask_and_scale=False, **kwargs))

    @classmethod
    def check_dataset(cls, dataset: xr.Dataset) -> Optional[int]:
        """
        A UGrid dataset needs a global attribute of Conventions = 'UGRID/...',
        and a variable with attribute cf_role = 'mesh_topology'
        and topology_dimension = 2
        """
        conventions = str(dataset.attrs.get('Conventions', ''))
        if not conventions.startswith('UGRID'):
            return None

        topology = Mesh2DTopology(dataset)
        try:
            mesh = topology.mesh_variable
        except ValueError:
            return None

        if mesh.attrs.get('topology_dimension') != 2:
            return None

        return Specificity.HIGH

    @classmethod
    def check_validity(cls, dataset: xr.Dataset) -> None:
        """Checks that the dataset is OK to use.
        Called during __init__, and raises exceptions if the dataset has problems.
        """
        # xarray does not handle masked integer variables well. It converts
        # them to doubles and masks them with nan. This plays havok with some
        # formats, such as UGRID with its integer topology variables.
        #
        # By opening a dataset with `mask_and_scale=False`, this is avoided,
        # but we have to take care of all masking ourselves, which is a drag.
        #
        # There is not a reliable way of detecting if a dataset has been opened
        # with `mask_and_scale=True` or `False`. The best heuristic is whether
        # these attributes are set in `attrs` or `encoding`. The masking and
        # scaling methods move these attributes to `encoding` as part of the
        # decoding process.
        #
        # See also: :func:`utils.mask_and_scale`.
        mask_and_scale_keys = ['_FillValue', 'missing_value', 'scale_factor', 'add_offset']
        masked_and_scaled = any(
            key in variable.encoding
            for key in mask_and_scale_keys
            for variable in dataset.variables.values()
        )
        if masked_and_scaled:
            raise Exception('\n'.join([
                "UGRID datasets must be opened with `mask_and_scale=False`",
                "    dataset = xarray.open_dataset(path, mask_and_scale=False)",
            ]))

    @cached_property
    def topology(self) -> Mesh2DTopology:
        """
        The :class:`Mesh2DTopology` of this dataset - which faces connect with which edges.
        """
        return Mesh2DTopology(self.dataset)

    def get_depth_name(self) -> str:
        return 'Mesh2_layers'

    def get_all_depth_names(self) -> List[str]:
        depths = ['Mesh2_layers']
        if 'Mesh2_layerfaces' in self.dataset.variables:
            depths.append('Mesh2_layerfaces')
        return depths

    def get_time_name(self) -> str:
        return 't'

    def ravel_index(self, index: UGridIndex) -> int:
        return index[1]

    def unravel_index(
        self,
        index: int,
        grid_kind: Optional[UGridKind] = None,
    ) -> UGridIndex:
        if grid_kind is None:
            grid_kind = UGridKind.face
        return (grid_kind, index)

    @cached_property
    def grid_kinds(self) -> FrozenSet[UGridKind]:
        items = [UGridKind.face, UGridKind.node]
        # The edge dimension is optional, not all UGRID datasets define it
        if self.topology.has_edge_dimension:
            items.append(UGridKind.edge)
        return frozenset(items)

    def get_grid_kind_and_size(
        self, data_array: xr.DataArray,
    ) -> Tuple[UGridKind, int]:
        if self.topology.face_dimension in data_array.dims:
            return (UGridKind.face, self.topology.face_count)
        if self.topology.has_edge_dimension and self.topology.edge_dimension in data_array.dims:
            return (UGridKind.edge, self.topology.edge_count)
        if self.topology.node_dimension in data_array.dims:
            return (UGridKind.node, self.topology.node_count)
        raise ValueError("Data array did not have any face, edge, or node dimension")

    @cached_property
    def polygons(self) -> np.ndarray:
        """Generate list of Polygons"""
        # X,Y coords of each node
        topology = self.topology
        node_x = topology.node_x.values
        node_y = topology.node_y.values
        faces = topology.face_node_connectivity
        fill_value = faces.attrs.get('_FillValue')

        def create_poly(face: np.ndarray) -> Polygon:
            vertices = face[face != fill_value]
            return Polygon(zip(node_x[vertices], node_y[vertices]))

        return np.array([create_poly(face) for face in faces.values], dtype=object)

    @cached_property
    def face_centres(self) -> np.ndarray:
        face_x, face_y = self.topology.face_x, self.topology.face_y
        if face_x is not None and face_y is not None:
            face_centres = np.column_stack((face_x, face_y))
            return cast(np.ndarray, face_centres)
        return super().face_centres

    def selector_for_index(self, index: UGridIndex) -> Dict[str, int]:
        kind, i = index
        if kind is UGridKind.face:
            return {self.topology.face_dimension: i}
        if kind is UGridKind.edge:
            if self.topology.has_edge_dimension:
                edge_dimension = str(self.topology.edge_dimension)
                return {edge_dimension: i}
            else:
                raise ValueError("Grid has no edge dimension")
        if kind is UGridKind.node:
            return {self.topology.node_dimension: i}
        raise ValueError("Invalid index")  # pragma: no-cover

    def make_linear(self, data_array: xr.DataArray) -> xr.DataArray:
        grid_kind, shape = self.get_grid_kind_and_size(data_array)
        grid_dimension = self.topology.dimension_for_grid_kind[grid_kind]
        return utils.linearise_dimensions(data_array, [grid_dimension])

    def make_clip_mask(self, clip_geometry: BaseGeometry) -> xr.Dataset:
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
        intersecting_face_indices = np.array([
            item.linear_index
            for polygon, item in self.spatial_index.query(clip_geometry)
            if polygon.intersects(clip_geometry)
        ])
        logger.debug("Found %d intersecting faces, adding buffer...", len(intersecting_face_indices))

        # Include all the neighbours of the intersecting faces
        included_face_indices = buffer_faces(intersecting_face_indices, self.topology)
        logger.debug("Total faces in mask: %d", len(included_face_indices))

        # Make a mask dataset
        return mask_from_face_indices(included_face_indices, self.topology)

    def apply_clip_mask(self, clip_mask: xr.Dataset, work_dir: Pathish) -> xr.Dataset:
        """
        Make a new dataset by applying a clip mask to this dataset.

        See also:
        * :meth:`make_clip_mask`
        * :func:`mask_from_face_indices`
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
        topology_variables: List[xr.DataArray] = [topology.mesh_variable]

        # This might be overly large for our new, smaller dataset, but that
        # does not matter.
        fill_value = topology.sensible_fill_value

        # This is the fill value used in the mask.
        new_fill_value = clip_mask.data_vars['new_node_index'].attrs['_FillValue']
        new_node_indices = clip_mask.data_vars['new_node_index'].values
        new_face_indices = clip_mask.data_vars['new_face_index'].values
        has_edges = 'new_edge_index' in clip_mask.data_vars
        if has_edges:
            new_edge_indices = clip_mask.data_vars['new_edge_index'].values

        def update_connectivity(
            connectivity: xr.DataArray,
            row_indices: np.ndarray,
            column_values: np.ndarray,
        ) -> xr.DataArray:
            logger.debug("Reindexing %r", connectivity.name)
            old_fill_value = connectivity.attrs.get('_FillValue', None)
            values = np.array([
                [column_values[item] if item != old_fill_value else fill_value for item in row]
                for row_index, row in enumerate(connectivity.values)
                if row_indices[row_index] != new_fill_value
            ], dtype=connectivity.dtype)
            return xr.DataArray(
                data=values, dims=connectivity.dims,
                name=connectivity.name, attrs={'_FillValue': fill_value}
            )

        # Re-index the face_node_connectivity variable
        topology_variables.append(update_connectivity(
            topology.face_node_connectivity,
            new_face_indices, new_node_indices))

        # Re-index each of the optional connectivity variables
        if has_edges and topology.has_valid_face_edge_connectivity:
            topology_variables.append(update_connectivity(
                topology.face_edge_connectivity,
                new_face_indices, new_edge_indices))

        if topology.has_valid_face_face_connectivity:
            topology_variables.append(update_connectivity(
                topology.face_face_connectivity,
                new_face_indices, new_face_indices))

        if has_edges and topology.has_valid_edge_face_connectivity:
            topology_variables.append(update_connectivity(
                topology.edge_face_connectivity,
                new_edge_indices, new_face_indices))

        if has_edges and topology.has_valid_edge_node_connectivity:
            topology_variables.append(update_connectivity(
                topology.edge_node_connectivity,
                new_edge_indices, new_node_indices))

        # Save all the topology variables to one combined dataset
        topology_path = work_path / (str(topology.mesh_variable.name) + ".nc")
        topology_dataset = xr.Dataset(
            data_vars={variable.name: variable for variable in topology_variables},
            coords=dataset.coords,
        )
        topology_dataset.to_netcdf(topology_path)
        mfdataset_paths.append(topology_path)

        topology_variable_names = list(topology_dataset.variables.keys())
        del topology_dataset
        del topology_variables

        logger.debug("Slicing data variables...")
        dimension_masks: Dict[Hashable, np.ndarray] = {
            topology.node_dimension: new_node_indices != new_fill_value,
            topology.face_dimension: new_face_indices != new_fill_value,
        }
        if has_edges:
            dimension_masks[topology.edge_dimension] = new_edge_indices != new_fill_value
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
                        slice_index = tuple([np.s_[:]] * index + [dimension_masks[dim]])  # type: ignore
                        values = values[slice_index]

                data_array = xr.DataArray(data=values, dims=data_array.dims, name=name)
                utils.disable_default_fill_value(data_array)
                data_array.to_netcdf(data_array_path)
                mfdataset_paths.append(data_array_path)

                del data_array
                del values

        logger.debug("Merging individual variables...")
        new_dataset = xr.open_mfdataset(mfdataset_paths, mask_and_scale=False, lock=False)
        return utils.dataset_like(dataset, new_dataset)
