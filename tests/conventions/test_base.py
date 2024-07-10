import dataclasses
import enum
import pathlib
from collections.abc import Hashable
from functools import cached_property

import numpy
import pandas
import pytest
import xarray
from matplotlib import pyplot
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from emsarray import masking, utils
from emsarray.conventions import Convention
from emsarray.exceptions import NoSuchCoordinateError
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
    def check_dataset(cls, dataset: xarray.Dataset) -> int | None:
        return None

    @cached_property
    def shape(self) -> tuple[int, int]:
        y, x = map(int, self.dataset['botz'].shape)
        return (y, x)

    @cached_property
    def grid_size(self) -> dict[SimpleGridKind, int]:
        return {SimpleGridKind.face: int(numpy.prod(self.shape))}

    def get_grid_kind(self, data_array: xarray.DataArray) -> SimpleGridKind:
        if set(data_array.dims) >= {'x', 'y'}:
            return SimpleGridKind.face
        raise ValueError("Unknown grid type")

    def get_all_geometry_names(self) -> list[Hashable]:
        return ['x', 'y']

    def wind_index(
        self,
        index: int,
        *,
        grid_kind: SimpleGridKind | None = None,
    ) -> SimpleGridIndex:
        y, x = map(int, numpy.unravel_index(index, self.shape))
        return SimpleGridIndex(y, x)

    def ravel_index(self, indexes: SimpleGridIndex) -> int:
        return int(numpy.ravel_multi_index((indexes.y, indexes.x), self.shape))

    def selector_for_indexes(
        self,
        indexes: list[SimpleGridIndex],
        *,
        index_dimension: Hashable | None = None,
    ) -> xarray.Dataset:
        if index_dimension is None:
            index_dimension = utils.find_unused_dimension(self.dataset, 'index')
        index_array = numpy.array([[index.x, index.y] for index in indexes])

        return xarray.Dataset({
            'x': (index_dimension, index_array[:, 0]),
            'y': (index_dimension, index_array[:, 1]),
        })

    def ravel(
        self,
        data_array: xarray.DataArray,
        *,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        self.get_grid_kind(data_array)
        return utils.ravel_dimensions(
            data_array, ['y', 'x'],
            linear_dimension=linear_dimension)

    def wind(
        self,
        data_array: xarray.DataArray,
        *,
        grid_kind: SimpleGridKind | None = None,
        axis: int | None = None,
        linear_dimension: Hashable | None = None,
    ) -> xarray.DataArray:
        if axis is not None:
            linear_dimension = data_array.dims[axis]
        elif linear_dimension is None:
            linear_dimension = data_array.dims[-1]

        return utils.wind_dimension(
            data_array,
            dimensions=['y', 'x'], sizes=self.shape,
            linear_dimension=linear_dimension)

    def drop_geometry(self) -> xarray.Dataset:
        return self.dataset

    @cached_property
    def polygons(self) -> numpy.ndarray:
        height, width = self.shape
        # Each polygon is a box from (x, y, x+1, y+1),
        # however the polygons around the edge are masked out with None.
        return numpy.array([
            box(x, y, x + 1, y + 1)
            if (0 < x < width - 1) and (0 < y < height - 1)
            else None
            for y in range(self.shape[0])
            for x in range(self.shape[1])
        ], dtype=numpy.object_)

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xarray.Dataset:
        if buffer > 0:
            raise ValueError("Can not buffer SimpleConvention clip masks")
        intersections = self.strtree.query(clip_geometry, predicate='intersects')
        cells = numpy.full(self.shape, False)
        cells[intersections] = True
        return xarray.Dataset({'cells': (['y', 'x'], cells)})

    def apply_clip_mask(self, clip_mask: xarray.Dataset, work_dir: Pathish) -> xarray.Dataset:
        return masking.mask_grid_dataset(self.dataset, clip_mask, work_dir)


def test_time_coordinate(datasets: pathlib.Path) -> None:
    dataset = xarray.open_dataset(datasets / 'times.nc')
    SimpleConvention(dataset).bind()
    time_coordinate = dataset.ems.time_coordinate
    assert time_coordinate.name == 'time'
    xarray.testing.assert_equal(time_coordinate, dataset['time'])


def test_time_coordinate_missing() -> None:
    dataset = xarray.Dataset()
    SimpleConvention(dataset).bind()
    with pytest.raises(NoSuchCoordinateError):
        dataset.ems.time_coordinate


@pytest.mark.parametrize('attrs', [
    {'positive': 'up'},
    {'positive': 'DOWN'},
    {'standard_name': 'depth'},
    {'axis': 'Z'},
], ids=lambda a: '{}:{}'.format(*next(iter(a.items()))))
def test_depth_coordinate(attrs: dict) -> None:
    dataset = xarray.Dataset({
        'name': (['dim'], [0, 1, 2], attrs),
    })
    SimpleConvention(dataset).bind()
    assert dataset.ems.depth_coordinate.name == 'name'
    xarray.testing.assert_equal(dataset.ems.depth_coordinate, dataset['name'])


def test_depth_coordinate_missing() -> None:
    dataset = xarray.Dataset()
    SimpleConvention(dataset).bind()
    with pytest.raises(NoSuchCoordinateError):
        dataset.ems.depth_coordinate


@pytest.mark.parametrize('include_time', [True, False])
@pytest.mark.parametrize('include_depth', [True, False])
def test_select_variables(
    include_time: bool,
    include_depth: bool,
):
    # Generate a dataset with some random data.
    # Time and depth dimensions are inluded or omitted based on the test arguments.
    generator = numpy.random.default_rng()
    expected_coords = {'y', 'x'}

    x_size, y_size = 5, 6
    dataset = xarray.Dataset({
        'x': (['x'], numpy.arange(x_size), {'units': 'degrees_east'}),
        'y': (['y'], numpy.arange(y_size), {'units': 'degrees_north'}),
        'colour': (['y', 'x'], numpy.arange(x_size * y_size).reshape((y_size, x_size)), {}),
        'flavour': (['y', 'x'], numpy.arange(x_size * y_size).reshape((y_size, x_size)), {}),
    })

    if include_time:
        expected_coords.add('time')
        dataset = dataset.assign_coords({
            'time': xarray.DataArray(
                dims=['time'],
                data=pandas.date_range('2023-08-01', '2023-08-24'),
            ),
        })
        dataset['time'].encoding['units'] = 'days since 1990-01-01 00:00:00 +10:00'
        time_size = dataset['time'].size
        dataset = dataset.assign({
            'eta': xarray.DataArray(
                dims=['time', 'y', 'x'],
                data=generator.uniform(-1.0, 1.0, (time_size, y_size, x_size)),
                attrs={'standard_name': 'sea_surface_height'},
            ),
        })

    if include_depth:
        expected_coords.add('depth')
        depth_size = 4
        dataset = dataset.assign_coords({
            'depth': xarray.DataArray(
                dims=['depth'],
                data=numpy.linspace(-10, 0, depth_size),
                attrs={'standard_name': 'depth', 'positive': 'up'},
            )
        })
        dataset = dataset.assign({
            'octarine': xarray.DataArray(
                dims=['depth', 'y', 'x'],
                data=(
                    generator.uniform(0.0, 1.0, (depth_size, y_size, x_size))
                    * numpy.linspace(100, 0, depth_size)[:, numpy.newaxis, numpy.newaxis]
                ),
                attrs={'standard_name': 'octarine_concentration'},
            ),
        })

    if include_depth and include_time:
        dataset = dataset.assign({
            'temperature': xarray.DataArray(
                dims=['time', 'depth', 'y', 'x'],
                data=(
                    generator.uniform(0, 3, (time_size, depth_size, y_size, x_size))
                    + numpy.linspace(2, 20, depth_size)[numpy.newaxis, :, numpy.newaxis, numpy.newaxis]
                )
            )
        })

    # Test various variable subset selections
    # It should be possible to select all sorts of subsets.
    # This should preserve coordinate information,
    # even if no variables using those coordinates are included in the subset.
    variable_choices = [{'colour'}]
    if include_time:
        variable_choices.append({'colour', 'eta'})
    if include_depth:
        variable_choices.append({'colour', 'octarine'})
    if include_depth and include_time:
        variable_choices.append({'colour', 'eta', 'octarine', 'temperature'})

    convention: Convention = SimpleConvention(dataset)
    for variables in variable_choices:
        subset = convention.select_variables(variables)
        expected_variables = variables | expected_coords
        assert set(subset.variables.keys()) == expected_variables
        for name in subset.variables.keys():
            xarray.testing.assert_equal(dataset[name], subset[name])


def test_mask():
    dataset = xarray.Dataset({
        'values': (['z', 'y', 'x'], numpy.random.standard_normal((5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)
    top_bottom = [False] * 20
    middle = [False] + [True] * 18 + [False]
    numpy.testing.assert_equal(
        convention.mask,
        numpy.array(top_bottom + middle * 8 + top_bottom),
    )


def test_geometry():
    dataset = xarray.Dataset({
        'values': (['z', 'y', 'x'], numpy.random.standard_normal((5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    # The geometry will be the union of all the polygons,
    # which results in some 'extra' vertices along the edge.
    assert convention.geometry == Polygon(
        [(1, x) for x in range(1, 10)]
        + [(y, 9) for y in range(2, 20)]
        + [(19, x) for x in reversed(range(1, 9))]
        + [(y, 1) for y in reversed(range(1, 19))]
    )


def test_bounds():
    dataset = xarray.Dataset({
        'values': (['z', 'y', 'x'], numpy.random.standard_normal((5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    assert convention.bounds == (1, 1, 19, 9)


def test_strtree():
    dataset = xarray.Dataset({
        'values': (['z', 'y', 'x'], numpy.random.standard_normal((5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    line = LineString([(-1, -1), (1.5, 1.5), (1.5, 2.5), (3.9, 3.9)])
    expected_intersections = {convention.ravel_index(index) for index in [
        SimpleGridIndex(1, 1), SimpleGridIndex(2, 1), SimpleGridIndex(2, 2),
        SimpleGridIndex(3, 2), SimpleGridIndex(3, 3)]}

    # Query the spatial index
    items = convention.strtree.query(line, predicate='intersects')
    assert set(items) == expected_intersections


def test_get_index_for_point_centre():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    # Polygons for this simple convetion is box(x, y, x+1, y+1):w
    index = convention.get_index_for_point(Point(1.5, 1.5))
    assert index.index == SimpleGridIndex(1, 1)

    index = convention.get_index_for_point(Point(2.5, 2.5))
    assert index.index == SimpleGridIndex(2, 2)


def test_get_index_for_point_vertex():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    point = Point(2, 2)

    # There should be four cells intersecting this point
    intersections = convention.strtree.query(point, predicate='intersects')
    assert len(intersections) == 4

    # `get_index_for_point` should still return a single item in this case
    index = convention.get_index_for_point(point)
    assert index.index == SimpleGridIndex(1, 1)

    # The point should be the first of the hits in index order
    assert index.linear_index == min(intersections)


def test_get_index_for_point_miss():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    point = Point(-1, -1)

    assert convention.get_index_for_point(point) is None


def test_selector_for_point():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)
    selector = convention.selector_for_index(SimpleGridIndex(3, 5))
    assert selector == {'x': 5, 'y': 3}


def test_select_index():
    # These tests can not be 100% comprehensive, as the simple convention only has
    # a single grid kind. The SHOC and UGRID convention tests will test the rest.
    dataset = xarray.Dataset(
        data_vars={
            'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
            'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
        },
        coords={
            'x': ('x', numpy.arange(20)),
            'y': ('y', numpy.arange(10)),
        },
    )
    convention = SimpleConvention(dataset)

    x, y = (3, 5)
    ds_point = convention.select_index(SimpleGridIndex(y, x))

    # The x and y dims should have been dropped, as they are now of size 1
    assert ds_point.sizes == {'t': 5, 'z': 5}
    # The x and y coords should be single values
    assert ds_point.coords['x'].values == x
    assert ds_point.coords['y'].values == y

    # temp should now be a single vertical column that varies over time
    assert ds_point['temp'].dims == ('t', 'z')
    assert ds_point['temp'].shape == (5, 5)
    numpy.testing.assert_equal(ds_point['temp'].values, dataset['temp'].values[:, :, y, x])

    # botz should be the depth at this single point
    assert ds_point['botz'].dims == ()
    assert ds_point['botz'].shape == ()
    assert ds_point['botz'].values == dataset['botz'].values[y, x]


def test_select_point():
    dataset = xarray.Dataset(
        data_vars={
            'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
            'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
        },
        coords={
            'x': ('x', numpy.arange(20)),
            'y': ('y', numpy.arange(10)),
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
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)
    convention.bind()

    # Naming a simple variable should work fine
    convention.plot('botz')
    pyplot.show.assert_called_once()
    pyplot.show.reset_mock()

    # This should raise an error, as 'temp' is 3D + time
    temp = dataset.data_vars['temp']
    with pytest.raises(ValueError):
        convention.plot(temp)
    pyplot.show.assert_not_called()
    pyplot.show.reset_mock()

    # Slice off the surface, at one time point
    initial_surface_temp = temp.isel({'z': 0, 't': 0})
    convention.plot(initial_surface_temp)
    pyplot.show.assert_called_once()
    pyplot.show.reset_mock()


def test_face_centres():
    # Test the fallback face_centres, which computes centres from polygon centroids
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    polygons = convention.polygons
    face_centres = convention.face_centres

    # Check a valid cell has a centre matching the polygon centroid
    i = 21
    assert polygons[i] is not None
    x, y = polygons[i].centroid.coords[0]
    numpy.testing.assert_equal(face_centres[i], [x, y])

    # Check a hole has a centre of [nan, nan]
    i = 0
    assert len(face_centres) == len(polygons)
    assert polygons[i] is None
    numpy.testing.assert_equal(face_centres[i], [numpy.nan, numpy.nan])


@pytest.mark.matplotlib
def test_make_poly_collection():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    patches = convention.make_poly_collection(cmap='plasma', edgecolor='black')
    assert len(patches.get_paths()) == len(convention.polygons[convention.mask])
    assert patches.get_cmap().name == 'plasma'
    # Colours get transformed in to RGBA arrays
    numpy.testing.assert_equal(patches.get_edgecolor(), [[0., 0., 0., 1.0]])


def test_make_poly_collection_data_array():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    patches = convention.make_poly_collection(data_array='botz')
    assert len(patches.get_paths()) == len(convention.polygons[convention.mask])

    values = convention.ravel(dataset.data_vars['botz'])[convention.mask]
    numpy.testing.assert_equal(patches.get_array(), values)
    assert patches.get_clim() == (numpy.nanmin(values), numpy.nanmax(values))


def test_make_poly_collection_data_array_and_array():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    array = numpy.random.standard_normal(len(convention.polygons[convention.mask]))

    with pytest.raises(TypeError):
        # Passing both array and data_array is a TypeError
        convention.make_poly_collection(data_array='botz', array=array)


def test_make_poly_collection_data_array_and_clim():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    # You can override the default clim if you want
    patches = convention.make_poly_collection(data_array='botz', clim=(-12, -8))
    assert patches.get_clim() == (-12, -8)


def test_make_poly_collection_data_array_dimensions():
    dataset = xarray.Dataset({
        'temp': (['t', 'z', 'y', 'x'], numpy.random.standard_normal((5, 5, 10, 20))),
        'botz': (['y', 'x'], numpy.random.standard_normal((10, 20)) - 10),
    })
    convention = SimpleConvention(dataset)

    with pytest.raises(ValueError):
        # temp needs subsetting first, so this should raise an error
        convention.make_poly_collection(data_array='temp')

    # One way to avoid this is to isel the data array
    convention.make_poly_collection(data_array=dataset.data_vars['temp'].isel(z=0, t=0))

    # Another way to avoid this is to isel the dataset
    dataset_0 = dataset.isel(z=0, t=0)
    convention = SimpleConvention(dataset_0)
    convention.make_poly_collection(data_array='temp')
