"""
Curvilinear Arakawa C grids.

See Also
--------
`Arakawa grids <https://en.wikipedia.org/wiki/Arakawa_grids>`_ on Wikipedia

"""
from __future__ import annotations

import enum
import logging
from functools import cached_property
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, cast

import numpy
import xarray
from shapely.geometry.base import BaseGeometry
from xarray.core.dataset import DatasetCoordinates

from emsarray import masking, utils
from emsarray.types import Pathish

from ._base import DimensionConvention, Specificity

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
        self, dataset: xarray.Dataset, *, latitude: Hashable, longitude: Hashable
    ) -> None:
        self.dataset = dataset
        self.latitude_name = latitude
        self.longitude_name = longitude

    @cached_property
    def latitude(self) -> xarray.DataArray:
        """The latitude :class:`~xarray.DataArray` coordinate variable."""
        return self.dataset[self.latitude_name]

    @cached_property
    def longitude(self) -> xarray.DataArray:
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
        return cast(int, numpy.prod(self.shape))

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


class ArakawaC(DimensionConvention[ArakawaCGridKind, ArakawaCIndex]):
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
        dataset: xarray.Dataset,
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

    @classmethod
    def check_dataset(cls, dataset: xarray.Dataset) -> Optional[int]:
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

    @cached_property
    def grid_dimensions(self) -> Dict[ArakawaCGridKind, Sequence[Hashable]]:
        return {
            kind: cast(Tuple[Hashable, Hashable], self.dataset[coordinates[0]].dims)
            for kind, coordinates in self.coordinate_names.items()
        }

    def unpack_index(self, index: ArakawaCIndex) -> Tuple[ArakawaCGridKind, Sequence[int]]:
        return index[0], index[1:]

    def pack_index(self, grid_kind: ArakawaCGridKind, indices: Sequence[int]) -> ArakawaCIndex:
        return cast(ArakawaCIndex, (grid_kind, *indices))

    @cached_property
    @utils.timed_func
    def polygons(self) -> numpy.ndarray:
        # Make an array of shape (j, i, 2) of all the nodes
        grid = numpy.stack([self.node.longitude.values, self.node.latitude.values], axis=-1)

        # Transform this in to an array of shape (topology.size, 4, 2)
        points = numpy.stack([
            grid[:-1, :-1],
            grid[:-1, +1:],
            grid[+1:, +1:],
            grid[+1:, :-1],
        ], axis=2).reshape((-1, 4, 2))

        return utils.make_polygons_with_holes(points)

    @cached_property
    def face_centres(self) -> numpy.ndarray:
        centres = numpy.column_stack((
            self.ravel(self.face.longitude).values,
            self.ravel(self.face.latitude).values,
        ))
        return cast(numpy.ndarray, centres)

    def get_all_geometry_names(self) -> List[Hashable]:
        return [
            self.face.longitude.name,
            self.face.latitude.name,
            self.node.longitude.name,
            self.node.latitude.name,
            self.left.longitude.name,
            self.left.latitude.name,
            self.back.longitude.name,
            self.back.latitude.name,
        ]

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xarray.Dataset:
        """
        Generate an xarray.Dataset with a mask for the cell centres, left and
        back faces, and nodes. The mask values will be True where the
        coordinate intersects the clip polygon, and False elsewhere.
        """
        logger.info("Finding intersecting cells for centre mask")

        # A cell is included if it intersects the clip polygon
        intersecting_indices = self.strtree.query(clip_geometry, predicate='intersects')
        face_mask = numpy.full(self.face.shape, fill_value=False)
        face_mask.ravel()[intersecting_indices] = True

        # Expand the mask by one cell around the clipped region, as a buffer
        if buffer > 0:
            face_mask = masking.blur_mask(face_mask, size=buffer)

        # Complete the rest of the mask
        grid_dimensions = cast(
            Dict[ArakawaCGridKind, Tuple[Hashable, Hashable]],
            self.grid_dimensions)
        return c_mask_from_centres(face_mask, grid_dimensions, self.dataset.coords)

    def apply_clip_mask(self, clip_mask: xarray.Dataset, work_dir: Pathish) -> xarray.Dataset:
        return masking.mask_grid_dataset(self.dataset, clip_mask, work_dir)


def c_mask_from_centres(
    face_mask: numpy.ndarray,
    dimensions: ArakawaCDimensions,
    coords: Optional[DatasetCoordinates] = None,
) -> xarray.Dataset:
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

    return xarray.Dataset(
        data_vars={
            "face_mask": xarray.DataArray(face_mask, dims=dimensions[ArakawaCGridKind.face]),
            "back_mask": xarray.DataArray(back_mask, dims=dimensions[ArakawaCGridKind.back]),
            "left_mask": xarray.DataArray(left_mask, dims=dimensions[ArakawaCGridKind.left]),
            "node_mask": xarray.DataArray(node_mask, dims=dimensions[ArakawaCGridKind.node]),
        },
        coords=coords,
    )
