"""
Curvilinear Arakawa C grids.

See also
--------
`Arakawa grids <https://en.wikipedia.org/wiki/Arakawa_grids>`_ on Wikipedia

"""
from __future__ import annotations

import enum
import logging
from functools import cached_property
from typing import Dict, Hashable, Optional, Tuple, cast

import numpy as np
import xarray as xr
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon, orient
from xarray.core.dataset import DatasetCoordinates

from emsarray import masking, utils
from emsarray.types import Pathish

from ._base import Format
from ._helpers import Specificity

logger = logging.getLogger(__name__)


class ArakawaCGridTopology:
    """
    A topology helper to deal with grid coordinates.

    Parameters
    ----------
    dataset : xarray.Dataset
        The Arakawa C gridded :class:`~xarray.Dataset` to introspect.
    latitude : Hashable
        The name of the latitude coordinate variable.
    longitude : Hashable
        The name of the longitude coordinate variable.
    """
    def __init__(
        self, dataset: xr.Dataset, *, latitude: Hashable, longitude: Hashable
    ) -> None:
        self.dataset = dataset
        self.latitude_name = latitude
        self.longitude_name = longitude

    @cached_property
    def latitude(self) -> xr.DataArray:
        """The latitude :class:`~xarray.DataArray` coordinate variable."""
        return self.dataset[self.latitude_name]

    @cached_property
    def longitude(self) -> xr.DataArray:
        """The logitude :class:`~xarray.DataArray` coordinate variable."""
        return self.dataset[self.longitude_name]

    @cached_property
    def j_dimension(self) -> Hashable:
        """The name of the ``j`` dimension for this grid kind."""
        return self.latitude.dims[0]

    @cached_property
    def i_dimension(self) -> Hashable:
        """The name of the ``i`` dimension for this grid kind."""
        return self.latitude.dims[1]

    @cached_property
    def shape(self) -> Tuple[int, int]:
        """The shape of this grid, as a tuple of ``(j, i)``."""
        return (self.dataset.dims[self.j_dimension], self.dataset.dims[self.i_dimension])

    @cached_property
    def size(self) -> int:
        """The size of this grid, ``j * i``."""
        return cast(int, np.prod(self.shape))

    def __repr__(self) -> str:
        bits = (f"{key}: {value!r}" for key, value in [
            ('latitude', self.latitude_name),
            ('longitude', self.longitude_name),
            ('shape', self.shape),
        ])
        details = ', '.join(bits)
        name = type(self).__name__
        return f'<{name}: {details}>'


class ArakawaCGridKind(str, enum.Enum):
    """Araawa C grid datasets can store data on
    cell faces, left edges, back edges, and nodes.
    The kind of grid is specified by this enum.
    """
    #: The face grid.
    #:
    #: :meta hide-value:
    face = 'face'
    #: The left edge grid.
    #: The left edge *i* dimension is one larger than for the face *i* dimension.
    #: A face ``(j, i)`` will have left edges ``(j, i)`` and ``(j, i + 1)``
    #:
    #: :meta hide-value:
    left = 'left'
    #: The back edge grid.
    #: The back edge *j* dimension is one larger than for the face *j* dimension.
    #: A face ``(j, i)`` will have back edges ``(j, i)`` and ``(j + 1, i)``
    #:
    #: :meta hide-value:
    back = 'back'
    #: The node grid.
    #: The node *i* and *j* dimensions are one larger than for the face *i* and *j* dimensions.
    #: A face ``(j, i)`` will have nodes ``(j, i)``, ``(j + 1, i)``, ``(j, i + 1``, and ``(j + 1, i + 1)``
    #:
    #: :meta hide-value:
    node = 'node'

    def __call__(self, j: int, i: int) -> ArakawaCIndex:
        return (self, j, i)


#: The native index type for Arakawa C grids
#: is a tuple with three elements: ``(kind, j, i).``
#:
#: :meta hide-value:
ArakawaCIndex = Tuple[ArakawaCGridKind, int, int]
ArakawaCCoordinates = Dict[ArakawaCGridKind, Tuple[Hashable, Hashable]]
ArakawaCDimensions = Dict[ArakawaCGridKind, Tuple[Hashable, Hashable]]


class ArakawaC(Format[ArakawaCGridKind, ArakawaCIndex]):
    """
    An Arakawa C grid is a curvilinear orthogonal grid
    with data defined on grid faces, edges, and nodes.
    The edges are split in to left and back edges.

    There is a topology helper for each of the
    :attr:`ArakawaC.face`, :attr:`ArakawaC.left`, :attr:`.back`, and :attr:`.node` grids.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing the Arakawa C grid
    coordinate_names : dict
        A dict defining which coordinate variables define which grids.
        The keys are each of 'face', 'left', 'back', or 'node';
        and the values are a two-tuple of strings
        naming the *latitude* and *longitude* coordinate variables for that grid kind:

        .. code-block:: python

            coordinate_names = {
                'face': ('y_face', 'x_face'),
                'left': ('y_left', 'x_left'),
                'back': ('y_back', 'x_back'),
                'node': ('y_node', 'x_node'),
            }

    """
    grid_kinds = frozenset(ArakawaCGridKind)
    default_grid_kind = ArakawaCGridKind.face

    coordinate_names: ArakawaCCoordinates

    def __init__(
        self,
        dataset: xr.Dataset,
        *,
        coordinate_names: Optional[Dict[Hashable, Tuple[Hashable, Hashable]]] = None,
    ):
        super().__init__(dataset)

        if coordinate_names is not None:
            if set(coordinate_names.keys()) != set(ArakawaCGridKind):
                kinds = ", ".join(kind.name for kind in ArakawaCGridKind)
                raise ValueError(f"coordinate_names must have keys {kinds}")

            self.coordinate_names = {
                ArakawaCGridKind(kind): coords
                for kind, coords in coordinate_names.items()
            }

        # Check that coordinate_names has been defined.
        # It may have been hard coded on a subclass.
        if not hasattr(self, 'coordinate_names'):
            raise TypeError(
                "Grid coordinates have not been defined. "
                "You must either pass the `coordinate_names` parameter, "
                "or set it on a subclass."
            )

    @cached_property
    def _dimensions_for_grid_kind(self) -> ArakawaCDimensions:
        return {
            kind: cast(Tuple[Hashable, Hashable], self.dataset[coordinates[0]].dims)
            for kind, coordinates in self.coordinate_names.items()
        }

    @classmethod
    def check_dataset(cls, dataset: xr.Dataset) -> Optional[int]:
        if not hasattr(cls, 'coordinate_names'):
            return None

        if all(
            coord in dataset.variables
            for coords in cls.coordinate_names.values() for coord in coords
        ):
            return Specificity.HIGH

        return None

    @cached_property
    def _topology_for_grid_kind(self) -> Dict[ArakawaCGridKind, ArakawaCGridTopology]:
        return {
            kind: ArakawaCGridTopology(
                self.dataset,
                latitude=coords[0],
                longitude=coords[1],
            )
            for kind, coords in self.coordinate_names.items()
        }

    @cached_property
    def face(self) -> ArakawaCGridTopology:
        """
        Topology for the face grid,
        an instance of :class:`ArakawaCGridTopology`.
        """
        return self._topology_for_grid_kind[ArakawaCGridKind.face]

    @cached_property
    def left(self) -> ArakawaCGridTopology:
        """
        Topology for the left edge grid,
        an instance of :class:`ArakawaCGridTopology`.
        """
        return self._topology_for_grid_kind[ArakawaCGridKind.left]

    @cached_property
    def back(self) -> ArakawaCGridTopology:
        """
        Topology for the back edge grid,
        an instance of :class:`ArakawaCGridTopology`.
        """
        return self._topology_for_grid_kind[ArakawaCGridKind.back]

    @cached_property
    def node(self) -> ArakawaCGridTopology:
        """
        Topology for the node grid,
        an instance of :class:`ArakawaCGridTopology`.
        """
        return self._topology_for_grid_kind[ArakawaCGridKind.node]

    def unravel_index(
        self,
        index: int,
        grid_kind: Optional[ArakawaCGridKind] = None,
    ) -> ArakawaCIndex:
        if grid_kind is None:
            grid_kind = ArakawaCGridKind.face
        topology = self._topology_for_grid_kind[grid_kind]
        j, i = map(int, np.unravel_index(index, topology.shape))
        return (grid_kind, j, i)

    def ravel_index(self, indices: ArakawaCIndex) -> int:
        grid_kind, j, i = indices
        topology = self._topology_for_grid_kind[grid_kind]
        return int(np.ravel_multi_index((j, i), topology.shape))

    def get_grid_kind_and_size(self, data_array: xr.DataArray) -> Tuple[ArakawaCGridKind, int]:
        dims = set(data_array.dims)
        for grid_kind in ArakawaCGridKind:
            grid_kind_dims = self._dimensions_for_grid_kind[grid_kind]
            if dims.issuperset(grid_kind_dims):
                topology = self._topology_for_grid_kind[grid_kind]
                return grid_kind, topology.size

        raise ValueError("Data array did not match any known grids")

    @cached_property
    def polygons(self) -> np.ndarray:
        # Keep these as 2D so that we can easily map centre->node indices
        x_node = self.node.longitude.values
        y_node = self.node.latitude.values

        def cell(index: int) -> Polygon:
            """
            Construct a single Polygon cell based on the 4 node points
            surrounding given cell centre
            """
            # 1D plus unravel index seems to be 10x faster than 2D indices
            # or can also try and use the mask approach further below - but this
            # seems to be fast enough
            (_face, j, i) = self.unravel_index(index)
            v1 = x_node[j, i], y_node[j, i]
            v2 = x_node[j + 1, i], y_node[j + 1, i]
            v3 = x_node[j + 1, i + 1], y_node[j + 1, i + 1]
            v4 = x_node[j, i + 1], y_node[j, i + 1]

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
        polygons = list(map(cell, range(self.face.size)))
        return np.array(polygons, dtype=object)

    @cached_property
    def face_centres(self) -> np.ndarray:
        centres = np.column_stack((
            self.make_linear(self.face.longitude).values,
            self.make_linear(self.face.latitude).values,
        ))
        return cast(np.ndarray, centres)

    def selector_for_index(self, index: ArakawaCIndex) -> Dict[Hashable, int]:
        kind, j, i = index
        topology = self._topology_for_grid_kind[kind]
        return {topology.j_dimension: j, topology.i_dimension: i}

    def drop_geometry(self) -> xr.Dataset:
        variables = [
            self.face.longitude.name,
            self.face.latitude.name,
            self.node.longitude.name,
            self.node.latitude.name,
            self.left.longitude.name,
            self.left.latitude.name,
            self.back.longitude.name,
            self.back.latitude.name,
        ]
        return self.dataset.drop_vars(variables)

    def make_linear(self, data_array: xr.DataArray) -> xr.DataArray:
        kind, size = self.get_grid_kind_and_size(data_array)
        topology = self._topology_for_grid_kind[kind]
        dimensions = [topology.j_dimension, topology.i_dimension]
        return utils.linearise_dimensions(data_array, list(dimensions))

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xr.Dataset:
        """
        Generate an xarray.Dataset with a mask for the cell centres, left and
        back faces, and nodes. The mask values will be True where the
        coordinate intersects the clip polygon, and False elsewhere.
        """
        logger.info("Finding intersecting cells for centre mask")

        # A cell is included if it intersects the clip polygon
        intersecting_indices = [
            item.linear_index
            for polygon, item in self.spatial_index.query(clip_geometry)
            if polygon.intersects(clip_geometry)]
        face_mask = np.full(self.face.shape, fill_value=False)
        face_mask.ravel()[intersecting_indices] = True

        # Expand the mask by one cell around the clipped region, as a buffer
        if buffer > 0:
            face_mask = masking.blur_mask(face_mask, size=buffer)

        # Complete the rest of the mask
        return c_mask_from_centres(face_mask, self._dimensions_for_grid_kind, self.dataset.coords)

    def apply_clip_mask(self, clip_mask: xr.Dataset, work_dir: Pathish) -> xr.Dataset:
        return masking.mask_grid_dataset(self.dataset, clip_mask, work_dir)


def c_mask_from_centres(
    face_mask: np.ndarray,
    dimensions: ArakawaCDimensions,
    coords: Optional[DatasetCoordinates] = None,
) -> xr.Dataset:
    """
    Create a mask for a SHOC standard file given a mask array for the cell
    centres to include. The full mask will include the centres, edges, and nodes.
    """
    # If a cell intersects the clip region at all, it will be included. This means
    # we need to include the cell centres, and also the edges for that cell.
    # After finding the centre mask, we can carefully pad that array to find the
    # left edge mask, back edge mask, and node masks.

    # An edge is included if the cell on either side is included.
    left_mask = masking.smear_mask(face_mask, [False, True])
    back_mask = masking.smear_mask(face_mask, [True, False])

    # A node is included if any of the four surrounding cells are included.
    node_mask = masking.smear_mask(face_mask, [True, True])

    return xr.Dataset(
        data_vars={
            "face_mask": xr.DataArray(face_mask, dims=dimensions[ArakawaCGridKind.face]),
            "back_mask": xr.DataArray(back_mask, dims=dimensions[ArakawaCGridKind.back]),
            "left_mask": xr.DataArray(left_mask, dims=dimensions[ArakawaCGridKind.left]),
            "node_mask": xr.DataArray(node_mask, dims=dimensions[ArakawaCGridKind.node]),
        },
        coords=coords,
    )
