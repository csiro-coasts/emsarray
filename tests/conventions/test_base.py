from __future__ import annotations

import dataclasses
import enum
from functools import cached_property
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry

from emsarray import masking, utils
from emsarray.conventions import Convention, SpatialIndexItem
from emsarray.types import Pathish


class SimpleGridKind(str, enum.Enum):
    face = 'face'


@dataclasses.dataclass(frozen=True)
class SimpleGridIndex:
    y: int
    x: int

    def __repr__(self) -> str:
        return f'<SimpleGridIndex {self}>'

    def __str__(self) -> str:
        return f'({self.y}, {self.x})'


class SimpleConvention(Convention[SimpleGridKind, SimpleGridIndex]):
    grid_kinds = frozenset(SimpleGridKind)
    default_grid_kind = SimpleGridKind.face

    @classmethod
    def check_dataset(cls, dataset: xr.Dataset) -> Optional[int]:
        return None

    def get_time_name(self) -> Hashable:
        return 't'

    def get_depth_name(self) -> Hashable:
        return 'z'

    def get_all_depth_names(self) -> List[Hashable]:
        return [self.get_depth_name()]

    @cached_property
    def shape(self) -> Tuple[int, int]:
        y, x = map(int, self.dataset['botz'].shape)
        return (y, x)

    def get_grid_kind_and_size(self, data_array: xr.DataArray) -> Tuple[SimpleGridKind, int]:
        expected = {'y', 'x'}
        if expected.issubset(data_array.dims):
            return (SimpleGridKind.face, int(np.prod(self.shape)))
        raise ValueError("Invalid dimensions")

    def unravel_index(
        self,
        index: int,
        grid_kind: Optional[SimpleGridKind] = None,
    ) -> SimpleGridIndex:
        y, x = map(int, np.unravel_index(index, self.shape))
        return SimpleGridIndex(y, x)

    def ravel_index(self, indices: SimpleGridIndex) -> int:
        return int(np.ravel_multi_index((indices.y, indices.x), self.shape))

    def selector_for_index(self, index: SimpleGridIndex) -> Dict[Hashable, int]:
        return {'x': index.x, 'y': index.y}

    def make_linear(self, data_array: xr.DataArray) -> xr.DataArray:
        return utils.linearise_dimensions(data_array, ['y', 'x'])

    def drop_geometry(self) -> xr.Dataset:
        return self.dataset

    @cached_property
    def polygons(self) -> np.ndarray:
        height, width = self.shape
        return np.array([
            box(x, y, x + 1, y + 1)
            if (0 < x < width - 1) and (0 < y < height - 1)
            else None
            for y in range(self.shape[0])
            for x in range(self.shape[1])
        ], dtype=np.object_)

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xr.Dataset:
        if buffer > 0:
            raise ValueError("Can not buffer SimpleConvention clip masks")
        intersections = [
            item.linear_index
            for polygon, item in self.spatial_index.query(clip_geometry)
            if item.polygon.intersects(clip_geometry)
        ]
        cells = np.full(self.shape, False)
        cells[intersections] = True
        return xr.Dataset({'cells': (['y', 'x'], cells)})

    def apply_clip_mask(self, clip_mask: xr.Dataset, work_dir: Pathish) -> xr.Dataset:
        return masking.mask_grid_dataset(self.dataset, clip_mask, work_dir)


def test_mask():
    dataset = xr.Dataset({
        'values': (['z', 'y', 'x'], np.random.standard_normal((5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)
    top_bottom = [False] * 20
    middle = [False] + [True] * 18 + [False]
    np.testing.assert_equal(
        convention.mask,
        np.array(top_bottom + middle * 8 + top_bottom),
    )


def test_spatial_index():
    dataset = xr.Dataset({
        'values': (['z', 'y', 'x'], np.random.standard_normal((5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    line = LineString([(-1, -1), (1.5, 1.5), (1.5, 2.5), (3.9, 3.9)])
    expected_intersections = {
        SimpleGridIndex(1, 1), SimpleGridIndex(2, 1), SimpleGridIndex(2, 2),
        SimpleGridIndex(3, 2), SimpleGridIndex(3, 3)}

    # Query the spatial index
    items = convention.spatial_index.query(line)['data']

    # The exact number of items returned isn't relevant, it should just be at
    # least the total expected interesections
    assert len(items) >= len(expected_intersections)

    # Each item should be a SpatialIndexItem
    assert isinstance(items[0], SpatialIndexItem)

    # We should be able to refine the match further
    actual_intersections = {
        item.index for item in items
        if item.polygon.intersects(line)}
    assert actual_intersections == expected_intersections


def test_get_index_for_point_centre():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    # Polygons for this simple convetion is box(x, y, x+1, y+1):w
    index = convention.get_index_for_point(Point(1.5, 1.5))
    assert index.index == SimpleGridIndex(1, 1)

    index = convention.get_index_for_point(Point(2.5, 2.5))
    assert index.index == SimpleGridIndex(2, 2)


def test_get_index_for_point_vertex():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    point = Point(2, 2)

    # There should be four cells intersecting this point
    intersections = [
        item for polygon, item in convention.spatial_index.query(point)
        if polygon.intersects(point)
    ]
    assert len(intersections) == 4

    # `get_index_for_point` should still return a single item in this case
    index = convention.get_index_for_point(point)
    assert index.index == SimpleGridIndex(1, 1)

    # The point should be the first of the hits in index order
    assert index.linear_index == min(hit.linear_index for hit in intersections)


def test_get_index_for_point_miss():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    point = Point(-1, -1)

    assert convention.get_index_for_point(point) is None


def test_selector_for_point():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)
    selector = convention.selector_for_index(SimpleGridIndex(3, 5))
    assert selector == {'x': 5, 'y': 3}


def test_select_index():
    # These tests can not be 100% comprehensive, as the simple convention only has
    # a single grid kind. The SHOC and UGRID convention tests will test the rest.
    dataset = xr.Dataset(
        data_vars={
            'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
            'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
        },
        coords={
            'x': ('x', np.arange(20)),
            'y': ('y', np.arange(10)),
        },
    )
    convention = SimpleConvention(dataset)

    x, y = (3, 5)
    ds_point = convention.select_index(SimpleGridIndex(y, x))

    # The x and y dims should have been dropped, as they are now of size 1
    assert ds_point.dims == {'t': 5, 'z': 5}
    # The x and y coords should be single values
    assert ds_point.coords['x'].values == x
    assert ds_point.coords['y'].values == y

    # temp should now be a single vertical column that varies over time
    assert ds_point['temp'].dims == ('t', 'z')
    assert ds_point['temp'].shape == (5, 5)
    np.testing.assert_equal(ds_point['temp'].values, dataset['temp'].values[:, :, y, x])

    # botz should be the depth at this single point
    assert ds_point['botz'].dims == ()
    assert ds_point['botz'].shape == ()
    assert ds_point['botz'].values == dataset['botz'].values[y, x]


def test_select_point():
    dataset = xr.Dataset(
        data_vars={
            'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
            'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
        },
        coords={
            'x': ('x', np.arange(20)),
            'y': ('y', np.arange(10)),
        },
    )
    convention = SimpleConvention(dataset)

    # The underlying implementation does this, and the individual methods are
    # tested in detail elsewhere
    point = Point(3.5, 5.5)
    index = convention.get_index_for_point(point).index
    expected = convention.select_index(index)
    actual = convention.select_point(point)
    assert actual == expected


@pytest.mark.matplotlib
def test_plot():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    # Naming a simple variable should work fine
    convention.plot('botz')

    # This should raise an error, as 'temp' is 3D + time
    temp = dataset.data_vars['temp']
    with pytest.raises(ValueError):
        convention.plot(temp)

    # Slice off the surface, at one time point
    initial_surface_temp = temp.isel({'z': 0, 't': 0})
    convention.plot(initial_surface_temp)


def test_face_centres():
    # Test the fallback face_centres, which computes centres from polygon centroids
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    polygons = convention.polygons
    face_centres = convention.face_centres

    # Check a valid cell has a centre matching the polygon centroid
    i = 21
    assert polygons[i] is not None
    x, y = polygons[i].centroid.coords[0]
    np.testing.assert_equal(face_centres[i], [x, y])

    # Check a hole has a centre of [nan, nan]
    i = 0
    assert len(face_centres) == len(polygons)
    assert polygons[i] is None
    np.testing.assert_equal(face_centres[i], [np.nan, np.nan])


@pytest.mark.matplotlib
def test_make_patch_collection():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    patches = convention.make_patch_collection(cmap='plasma', edgecolor='black')
    assert len(patches.get_paths()) == len(convention.polygons[convention.mask])
    assert patches.get_cmap().name == 'plasma'
    # Colours get transformed in to RGBA arrays
    np.testing.assert_equal(patches.get_edgecolor(), [[0., 0., 0., 1.0]])


def test_make_patch_collection_data_array():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    patches = convention.make_patch_collection(data_array='botz')
    assert len(patches.get_paths()) == len(convention.polygons[convention.mask])

    values = convention.make_linear(dataset.data_vars['botz'])[convention.mask]
    np.testing.assert_equal(patches.get_array(), values)
    assert patches.get_clim() == (np.nanmin(values), np.nanmax(values))


def test_make_patch_collection_data_array_and_array():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    array = np.random.standard_normal(len(convention.polygons[convention.mask]))

    with pytest.raises(TypeError):
        # Passing both array and data_array is a TypeError
        convention.make_patch_collection(data_array='botz', array=array)


def test_make_patch_collection_data_array_and_clim():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    # You can override the default clim if you want
    patches = convention.make_patch_collection(data_array='botz', clim=(-12, -8))
    assert patches.get_clim() == (-12, -8)


def test_make_patch_collection_data_array_dimensions():
    dataset = xr.Dataset({
        'temp': (['t', 'z', 'y', 'x'], np.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], np.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    with pytest.raises(ValueError):
        # temp needs subsetting first, so this should raise an error
        convention.make_patch_collection(data_array='temp')

    # One way to avoid this is to isel the data array
    convention.make_patch_collection(data_array=dataset.data_vars['temp'].isel(z=0, t=0))

    # Another way to avoid this is to isel the dataset
    dataset_0 = dataset.isel(z=0, t=0)
    convention = SimpleConvention(dataset_0)
    convention.make_patch_collection(data_array='temp')
