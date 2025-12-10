"""
Curvilinear Arakawa C grids.

See Also
--------
`Arakawa grids <https://en.wikipedia.org/wiki/Arakawa_grids>`_ on Wikipedia

"""
import enum
import logging
from collections.abc import Hashable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy
import shapely
import xarray
from shapely.geometry.base import BaseGeometry
from xarray.core.dataset import DatasetCoordinates

from emsarray import masking, plot, utils
from emsarray.operations import triangulate
from emsarray.types import DataArrayOrName, Pathish

from ._base import DimensionConvention, Specificity

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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
    def shape(self) -> tuple[int, int]:
        """The shape of this grid, as a tuple of ``(j, i)``."""
        return (
            self.dataset.sizes[self.j_dimension],
            self.dataset.sizes[self.i_dimension],
        )

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


class ArakawaCGridKind(enum.StrEnum):
    """Arakawa C grid datasets can store data on
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


ArakawaCCoordinates = dict[ArakawaCGridKind, tuple[Hashable, Hashable]]
ArakawaCDimensions = dict[ArakawaCGridKind, tuple[Hashable, Hashable]]


class ArakawaC(DimensionConvention[ArakawaCGridKind]):
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
    geometry_types = {
        ArakawaCGridKind.face: shapely.Polygon,
        ArakawaCGridKind.left: shapely.LineString,
        ArakawaCGridKind.back: shapely.LineString,
        ArakawaCGridKind.node: shapely.Point,
    }

    coordinate_names: ArakawaCCoordinates

    def __init__(
        self,
        dataset: xarray.Dataset,
        *,
        coordinate_names: dict[ArakawaCGridKind, tuple[Hashable, Hashable]] | None = None,
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
    def check_dataset(cls, dataset: xarray.Dataset) -> int | None:
        if not hasattr(cls, 'coordinate_names'):
            return None

        if all(
            coord in dataset.variables
            for coords in cls.coordinate_names.values() for coord in coords
        ):
            return Specificity.HIGH

        return None

    @cached_property
    def _topology_for_grid_kind(self) -> dict[ArakawaCGridKind, ArakawaCGridTopology]:
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
    def grid_dimensions(self) -> dict[ArakawaCGridKind, Sequence[Hashable]]:
        return {
            kind: cast(tuple[Hashable, Hashable], self.dataset[coordinates[0]].dims)
            for kind, coordinates in self.coordinate_names.items()
        }

    def _make_geometry(self, grid_kind: ArakawaCGridKind) -> numpy.ndarray:
        if grid_kind is ArakawaCGridKind.face:
            return self._make_polygons()
        if grid_kind is ArakawaCGridKind.left:
            return self._make_left_edges()
        if grid_kind is ArakawaCGridKind.back:
            return self._make_back_edges()
        if grid_kind is ArakawaCGridKind.node:
            return self._make_nodes()
        raise ValueError(f"Invalid grid kind {grid_kind}")

    def _make_polygons(self) -> numpy.ndarray:
        j_size, i_size = self.face.shape
        longitude = self.node.longitude.values
        latitude = self.node.latitude.values

        # Preallocate the points array. We will copy data straight in to this
        # to save repeated memory allocations.
        points = numpy.empty(shape=(i_size, 4, 2), dtype=longitude.dtype)
        # Preallocate the output array so we can fill it in batches
        out = numpy.full(shape=self.face.size, fill_value=None, dtype=object)
        # Construct polygons row by row
        for j in range(j_size):
            points[:, 0, 0] = longitude[j + 0, :-1]
            points[:, 1, 0] = longitude[j + 0, +1:]
            points[:, 2, 0] = longitude[j + 1, +1:]
            points[:, 3, 0] = longitude[j + 1, :-1]

            points[:, 0, 1] = latitude[j + 0, :-1]
            points[:, 1, 1] = latitude[j + 0, +1:]
            points[:, 2, 1] = latitude[j + 1, +1:]
            points[:, 3, 1] = latitude[j + 1, :-1]

            j_slice = slice(j * i_size, (j + 1) * i_size)
            utils.make_polygons_with_holes(points, out=out[j_slice])

        return out

    def _make_left_edges(self) -> numpy.ndarray:
        j_size, i_size = self.left.shape
        longitude = self.node.longitude.values
        latitude = self.node.latitude.values

        points = numpy.empty(shape=(i_size, 2, 2), dtype=longitude.dtype)

        out = numpy.full(shape=self.left.size, fill_value=None, dtype=object)
        for j in range(j_size):
            points[:, 0, 0] = longitude[j + 0, :]
            points[:, 1, 0] = longitude[j + 1, :]

            points[:, 0, 1] = latitude[j + 0, :]
            points[:, 1, 1] = latitude[j + 1, :]

            j_slice = slice(j * i_size, (j + 1) * i_size)
            utils.make_linestrings_with_holes(points, out=out[j_slice])

        return out

    def _make_back_edges(self) -> numpy.ndarray:
        j_size, i_size = self.back.shape
        longitude = self.node.longitude.values
        latitude = self.node.latitude.values

        points = numpy.empty(shape=(i_size, 2, 2), dtype=longitude.dtype)

        out = numpy.full(shape=self.back.size, fill_value=None, dtype=object)
        for j in range(j_size):
            points[:, 0, 0] = longitude[j, :-1]
            points[:, 1, 0] = longitude[j, +1:]

            points[:, 0, 1] = latitude[j, :-1]
            points[:, 1, 1] = latitude[j, +1:]

            j_slice = slice(j * i_size, (j + 1) * i_size)
            utils.make_linestrings_with_holes(points, out=out[j_slice])

        return out

    def _make_nodes(self) -> numpy.ndarray:
        j_size, i_size = self.node.shape
        longitude = self.node.longitude.values
        latitude = self.node.latitude.values

        points = numpy.empty(shape=(i_size, 2), dtype=longitude.dtype)

        out = numpy.full(shape=self.node.size, fill_value=None, dtype=object)
        for j in range(j_size):
            points[:, 0] = longitude[j, :]
            points[:, 1] = latitude[j, :]

            j_slice = slice(j * i_size, (j + 1) * i_size)
            utils.make_points_with_holes(points, out=out[j_slice])

        return out

    def _make_geometry_centroid(self, grid_kind: ArakawaCGridKind) -> numpy.ndarray:
        if grid_kind is ArakawaCGridKind.face:
            topology = self.face
        elif grid_kind is ArakawaCGridKind.left:
            topology = self.left
        elif grid_kind is ArakawaCGridKind.back:
            topology = self.back
        elif grid_kind is ArakawaCGridKind.node:
            topology = self.node

        coords = numpy.column_stack((
            self.ravel(self.face.longitude).values,
            self.ravel(self.face.latitude).values,
        ))
        points = numpy.full(shape=topology.size, dtype=object, fill_value=None)
        valid_coords = numpy.flatnonzero(numpy.all(~numpy.isnan(coords), axis=1))
        shapely.points(coords[valid_coords], out=points, indices=valid_coords)
        return cast(numpy.ndarray, points)

    def get_all_geometry_names(self) -> list[Hashable]:
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
        face_grid = self.grids[ArakawaCGridKind.face]
        intersecting_indexes = face_grid.strtree.query(clip_geometry, predicate='intersects')
        face_mask_da = xarray.DataArray(numpy.full(face_grid.size, fill_value=False))
        face_mask_da.values[intersecting_indexes] = True
        face_mask = face_grid.wind(face_mask_da).values

        # Expand the mask by one cell around the clipped region, as a buffer
        if buffer > 0:
            face_mask = masking.blur_mask(face_mask, size=buffer)

        # Complete the rest of the mask
        grid_dimensions = cast(
            dict[ArakawaCGridKind, tuple[Hashable, Hashable]],
            self.grid_dimensions)
        return c_mask_from_centres(face_mask, grid_dimensions, self.dataset.coords)

    def apply_clip_mask(self, clip_mask: xarray.Dataset, work_dir: Pathish) -> xarray.Dataset:
        return masking.mask_grid_dataset(self.dataset, clip_mask, work_dir)

    def make_artist(
        self,
        axes: 'Axes',
        variable: DataArrayOrName | tuple[DataArrayOrName, ...],
        **kwargs: Any,
    ) -> 'plot.GridArtist':
        data_array = utils.names_to_data_arrays(self.dataset, variable)

        if isinstance(data_array, xarray.DataArray):
            grid_kind = self.get_grid_kind(data_array)
            if grid_kind is ArakawaCGridKind.face:
                return plot.artists.make_polygon_scalar_collection(
                    axes, self.grids[grid_kind], data_array, **kwargs)

            if grid_kind is ArakawaCGridKind.node:
                return plot.artists.make_node_scalar_artist(
                    axes, self.grids[grid_kind], data_array, **kwargs)

        else:
            grid_kinds = tuple(self.get_grid_kind(d) for d in data_array)
            if grid_kinds == (ArakawaCGridKind.face, ArakawaCGridKind.face):
                return plot.artists.make_polygon_vector_quiver(
                    axes, self.grids[ArakawaCGridKind.face], data_array, **kwargs)

        raise ValueError("I don't know how to plot this")

    def plot_geometry(
        self,
        axes: 'Axes',
    ) -> 'plot.GridArtist':
        grid = self.grids[ArakawaCGridKind.face]
        collection = plot.artists.PolygonScalarCollection.from_grid(
            grid,
            edgecolor='grey',
            facecolor='blue',
            linewidth=0.5,
        )
        axes.add_collection(collection)
        return collection

    def make_triangulation(self) -> triangulate.Triangulation:
        vertices = self.grids[ArakawaCGridKind.node].geometry
        polygons = self.grids[ArakawaCGridKind.face].geometry

        j_size, i_size = self.node.shape
        xx, yy = numpy.meshgrid(numpy.arange(i_size - 1), numpy.arange(j_size - 1) * i_size)
        bottom_left_indexes = (xx + yy).flatten()
        offsets = numpy.array([0, 1, i_size + 1, i_size])
        polygon_vertex_indexes = bottom_left_indexes[:, None].repeat(4, axis=1) + offsets

        vertex_coordinates, triangles, face_indexes = triangulate.triangulate(vertices, polygons, polygon_vertex_indexes)
        return triangulate.Triangulation[ArakawaCGridKind](
            vertices=vertex_coordinates,
            triangles=triangles,
            face_indexes=face_indexes,
            face_grid_kind=ArakawaCGridKind.face,
            vertex_grid_kind=ArakawaCGridKind.node)


def c_mask_from_centres(
    face_mask: numpy.ndarray,
    dimensions: ArakawaCDimensions,
    coords: DatasetCoordinates | None = None,
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
