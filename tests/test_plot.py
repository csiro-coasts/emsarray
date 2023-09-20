import cartopy.crs
import matplotlib.figure
import numpy
import pytest
import shapely

from emsarray.plot import add_landmarks, polygons_to_collection


@pytest.mark.matplotlib
def test_polygons_to_collection():
    polygons = [
        shapely.Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)])
        for i in range(10)
    ]
    data = numpy.random.random(10) * 10
    patch_collection = polygons_to_collection(
        polygons, array=data, cmap='autumn', clim=(0, 10))

    # Check that the polygons came through
    assert len(patch_collection.get_paths()) == 10

    # Check that keyword arguments were passed through
    assert patch_collection.get_cmap().name == 'autumn'
    numpy.testing.assert_equal(patch_collection.get_array(), data)


@pytest.mark.matplotlib
def test_add_landmarks():
    figure = matplotlib.figure.Figure()
    axes = figure.add_subplot(projection=cartopy.crs.PlateCarree())

    landmarks = [
        ('Origin', shapely.Point(0, 0)),
        ('São Tomé and Príncipe', shapely.Point(6.607735, 0.2633684)),
        ('San Antonio de Palé', shapely.Point(5.640007, -1.428858)),
        ('Bela Vista', shapely.Point(7.410149, 1.614794)),
        ('Bioko', shapely.Point(8.745365, 3.433421)),
    ]
    add_landmarks(axes, landmarks)

    assert len(landmarks) == len(axes.texts)
    for landmark, text in zip(landmarks, axes.texts):
        assert text.get_text() == landmark[0]
        assert text.xy == landmark[1].coords.xy
