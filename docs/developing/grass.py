# > imports
import enum
from functools import cached_property
from typing import Optional, Tuple, Dict

import numpy as np
import xarray as xr
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from emsarray.conventions import Convention, Specificity
from emsarray.masking import blur_mask
from emsarray.types import Pathish
from emsarray.utils import linearise_dimensions
# <


class GrassGridKind(enum.Enum):
    blade = 'blade'
    meadow = 'meadow'


GrassIndex = Tuple[GrassGridKind, int, int]


class Grass(Convention[GrassGridKind, GrassIndex]):

    #: All the grid kinds this dataset has
    grid_kinds = frozenset(GrassGridKind)

    #: Indicates the grid kind of cells
    default_grid_kind = GrassGridKind.blade

    @classmethod
    def check_dataset(cls, dataset: xr.Dataset) -> Optional[int]:
        # A Grass dataset is recognised by the 'Conventions' global attribute
        if dataset.attrs['Conventions'] == 'Grass 1.0':
            return Specificity.HIGH
        return None

    def ravel_index(self, index: GrassIndex) -> int:
        """Make a linear index from a native index"""
        kind, warp, weft = index
        # Meadows indexes are transposed from blade indexes
        if kind is GrassGridKind.meadow:
            return warp * self.dataset.dims['weft'] + weft
        else:
            return weft * self.dataset.dims['warp'] + warp

    def unravel_index(
        self,
        index: int,
        grid_kind: Optional[GrassGridKind] = None,
    ) -> GrassIndex:
        """Make a native index from a linear index"""
        grid_kind = grid_kind or self.default_grid_kind
        if grid_kind is GrassGridKind.meadow:
            warp, weft = divmod(index, self.dataset.dims['weft'])
        else:
            weft, warp = divmod(index, self.dataset.dims['warp'])
        return (grid_kind, warp, weft)

    def get_grid_kind_and_size(
        self, data_array: xr.DataArray,
    ) -> Tuple[GrassGridKind, int]:
        """
        For the given DataArray from this Dataset,
        find out what kind of grid it is, and the linear size of that grid.
        """
        size = self.dataset.dims['warp'] * self.dataset.dims['weft']
        if data_array.dims[-2:] == ('warp', 'weft'):
            return GrassGridKind.meadow, size
        if data_array.dims[-2:] == ('weft', 'warp'):
            return GrassGridKind.blade, size
        raise ValueError(
            "DataArray does not appear to be either a blade or meadow grid")

    def make_linear(self, data_array: xr.DataArray) -> xr.DataArray:
        """
        Make the given DataArray linear in its grid dimensions.
        """
        grid_kind, size = self.get_grid_kind_and_size(data_array)
        if grid_kind is GrassGridKind.meadow:
            dimensions = ['warp', 'weft']
        else:
            dimensions = ['weft', 'warp']
        return linearise_dimensions(data_array, dimensions)

    def selector_for_index(self, index: GrassIndex) -> Dict[str, int]:
        """
        Make a selector for a particular index.
        This selector can be passed to Dataset.isel().
        """
        kind, warp, weft = index
        return {'warp': warp, 'weft': weft}

    @cached_property
    def polygons(self) -> np.ndarray:
        def make_polygon_for_cell(warp: int, weft: int) -> Polygon:
            # Implementation left as an exercise for the reader
            return Polygon(...)

        return np.array([
            make_polygon_for_cell(warp, weft)
            for warp in range(self.dataset.dimensions['warp'])
            for weft in range(self.dataset.dimensions['weft'])
        ])

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xr.Dataset:
        # Find all the blades that intersect the clip geometry
        intersecting_blades = [
            item
            for item, polygon in self.spatial_index.query(clip_geometry)
            if polygon.intersects(clip_geometry)
        ]
        # Get all the linear indexes of the intersecting blades
        blade_indexes = np.array([i.linear_index for i in intersecting_blades])
        # Find all the meadows associated with each intesecting blade
        meadow_indexes = np.unique([
            self.ravel_index(blade_index)
            for item in intersecting_blades
            for blade_index in self.get_meadows_for_blade(item.index)
        ])

        warp = self.dataset.dims['warp']
        weft = self.dataset.dims['weft']

        # Make a 2d array of which blades to keep
        keep_blades = np.zeros((weft, warp), dtype=bool)
        keep_blades.ravel()[blade_indexes] = True

        # Same for meadows
        keep_meadows = np.zeros((warp, weft), dtype=bool)
        keep_meadows.ravel()[meadow_indexes] = True

        # Blur the masks a bit if the clip region needs buffering
        if buffer > 0:
            keep_blades = blur_mask(keep_blades, size=buffer)
            keep_meadows = blur_mask(keep_meadows, size=buffer)

        # Make a dataset out of these masks
        return xr.Dataset(
            data_vars={
                'blades': xr.DataArray(data=keep_blades, dims=['weft', 'warp']),
                'meadows': xr.DataArray(data=keep_meadows, dims=['warp', 'weft']),
            },
        )

    def apply_clip_mask(self, clip_mask: xr.Dataset, work_dir: Pathish) -> xr.Dataset:
        # You're on your own, here.
        # This depends entirely on how the mask and datasets interact.
        pass
