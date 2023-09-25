import logging
import pathlib

import cartopy.crs
import matplotlib.figure
import matplotlib.pyplot
import numpy
import pytest
import shapely

import emsarray
from emsarray.plot import add_landmarks, polygons_to_collection

logger = logging.getLogger(__name__)


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


@pytest.mark.matplotlib(mock_coast=True)
@pytest.mark.tutorial
def test_plot(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
):
    """
    Test plotting a variable with no long_name attribute works.
    Regression test for https://github.com/csiro-coasts/emsarray/issues/105
    """
    dataset = emsarray.tutorial.open_dataset('gbr4')
    temp = dataset['temp'].copy()
    temp = temp.isel(time=0, k=-1)

    dataset.ems.plot(temp)

    figure = matplotlib.pyplot.gcf()
    axes = figure.axes[0]
    assert axes.get_title() == 'Temperature\n2022-05-11T14:00:00.000000000'

    matplotlib.pyplot.savefig(tmp_path / 'plot.png')
    logger.info("Saved plot to %r", tmp_path / 'plot.png')


@pytest.mark.matplotlib(mock_coast=True)
@pytest.mark.tutorial
def test_plot_no_long_name(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
):
    """
    Test plotting a variable with no long_name attribute works.
    Regression test for https://github.com/csiro-coasts/emsarray/issues/105
    """
    dataset = emsarray.tutorial.open_dataset('gbr4')
    temp = dataset['temp'].copy()
    temp = temp.isel(time=0, k=-1)
    del temp.attrs['long_name']

    dataset.ems.plot(temp)

    figure = matplotlib.pyplot.gcf()
    axes = figure.axes[0]
    assert axes.get_title() == '2022-05-11T14:00:00.000000000'

    matplotlib.pyplot.savefig(tmp_path / 'plot.png')
    logger.info("Saved plot to %r", tmp_path / 'plot.png')
