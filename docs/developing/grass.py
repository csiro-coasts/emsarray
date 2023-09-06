# > imports
import enum
from functools import cached_property
from typing import Dict, Hashable, Optional, Sequence, Tuple

import numpy
import xarray
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from emsarray.conventions import DimensionConvention, Specificity
from emsarray.masking import blur_mask
from emsarray.types import Pathish
# <


class GrassGridKind(enum.Enum):
    field = 'field'
    fence = 'fence'


GrassIndex = Tuple[GrassGridKind, Sequence[int]]


class Grass(DimensionConvention[GrassGridKind, GrassIndex]):

    #: All the grid kinds this dataset has
    grid_kinds = frozenset(GrassGridKind)

    #: Indicates the grid kind of cells
    default_grid_kind = GrassGridKind.field

    @classmethod
    def check_dataset(cls, dataset: xarray.Dataset) -> Optional[int]:
        # A Grass dataset is recognised by the 'Conventions' global attribute
        if dataset.attrs['Conventions'] == 'Grass 1.0':
            return Specificity.HIGH
        return None

    def unpack_index(self, index: GrassIndex) -> Tuple[GrassGridKind, Sequence[int]]:
        return index[0], list(index[1])

    def pack_index(self, grid_kind: GrassGridKind, indices: Sequence[int]) -> GrassIndex:
        return (grid_kind, list(indices))

    @cached_property
    def grid_dimensions(self) -> Dict[GrassGridKind, Sequence[Hashable]]:
        return {
            GrassGridKind.field: ['warp', 'weft'],
            GrassGridKind.fence: ['post'],
        }

    @cached_property
    def polygons(self) -> numpy.ndarray:
        def make_polygon_for_cell(warp: int, weft: int) -> Polygon:
            # Implementation left as an exercise for the reader
            return Polygon(...)

        return numpy.array([
            make_polygon_for_cell(warp, weft)
            for warp in range(self.dataset.dimensions['warp'])
            for weft in range(self.dataset.dimensions['weft'])
        ])

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xarray.Dataset:
        # Find all the fields that intersect the clip geometry
        intersecting_fields = [
            field
            for field, polygon in self.spatial_index.query(clip_geometry)
            if polygon.intersects(clip_geometry)
        ]
        # Get all the linear indexes of the intersecting blades
        field_indexes = numpy.array([i.linear_index for i in intersecting_fields])
        # Find all the fences associated with each intesecting field
        fence_indexes = numpy.unique([
            self.ravel_index(fence_index)
            for field in intersecting_fields
            for fence_index in self.get_fences_around_field(field.index)
        ])

        # Make an array of which fields to keep
        keep_fields = xarray.DataArray(
            data=numpy.zeros(self.grid_size[GrassGridKind.field], dtype=bool),
            dims=['index'],
        )
        keep_fields.values[field_indexes] = True
        keep_fields = self.wind(keep_fields, grid_kind=GrassGridKind.field)

        # Same for fences
        keep_fences = xarray.DataArray(
            data=numpy.zeros(self.grid_size[GrassGridKind.fence], dtype=bool),
            dims=['index'],
        )
        keep_fences.values[fence_indexes] = True
        keep_fences = self.wind(keep_fences, grid_kind=GrassGridKind.fence)

        # Blur the masks a bit if the clip region needs buffering
        if buffer > 0:
            keep_fields.values = blur_mask(keep_fields.values, size=buffer)

        # Make a dataset out of these masks
        return xarray.Dataset(
            data_vars={
                'fields': keep_fields,
                'fences': keep_fences,
            },
        )

    def apply_clip_mask(self, clip_mask: xarray.Dataset, work_dir: Pathish) -> xarray.Dataset:
        # You're on your own, here.
        # This depends entirely on how the mask and datasets interact.
        pass
