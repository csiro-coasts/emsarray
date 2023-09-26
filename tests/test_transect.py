import logging
import pathlib

import matplotlib
import pytest
import shapely

import emsarray.transect

logger = logging.getLogger(__name__)


@pytest.mark.matplotlib(mock_coast=True)
@pytest.mark.tutorial
def test_plot(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
):
    dataset = emsarray.tutorial.open_dataset('gbr4')
    temp = dataset['temp'].copy()
    temp = temp.isel(time=-1)

    line = shapely.LineString([
        [152.9768944, -25.4827962],
        [152.9701996, -25.4420345],
        [152.9727745, -25.3967620],
        [152.9623032, -25.3517828],
        [152.9401588, -25.3103560],
        [152.9173279, -25.2538563],
        [152.8962135, -25.1942238],
        [152.8692627, -25.0706729],
        [152.8623962, -24.9698750],
        [152.8472900, -24.8415806],
        [152.8308105, -24.6470172],
        [152.7607727, -24.3521012],
        [152.6392365, -24.1906056],
        [152.4792480, -24.0615124],
    ])
    emsarray.transect.plot(
        dataset, line, temp,
        bathymetry=dataset['botz'])

    figure = matplotlib.pyplot.gcf()
    axes = figure.axes[0]
    # This is assembled from the variable long_name and the time coordinate
    assert axes.get_title() == 'Temperature\n2022-05-11T14:00'
    # This is the long_name of the depth coordinate
    assert axes.get_ylabel() == 'Z coordinate'
    # This is made up
    assert axes.get_xlabel() == 'Distance along transect'

    colorbar = figure.axes[-1]
    # This is the variable units
    assert colorbar.get_ylabel() == 'degrees C'

    matplotlib.pyplot.savefig(tmp_path / 'plot.png')
    logger.info("Saved plot to %r", tmp_path / 'plot.png')
