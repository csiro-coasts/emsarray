import pytest
import shapely
from matplotlib.figure import Figure

import emsarray
from emsarray.transect import Transect
from emsarray.transect.artists import CrossSectionArtist, TransectStepArtist
from emsarray.utils import estimate_bounds_1d

# Testing the intersection data of the transect is difficult
# without simply hard coding the expected intersections.
# Instead we mostly test that the transect functions return reasonably shaped things,
# and rely on the plot tests when comparing actual results.

# This line stays within the bounds of the GBR4 dataset.
LINE_INBOUNDS = shapely.LineString([
    [150.172296, -19.1132187],
    [150.985268, -20.7740421],
    [151.657324, -22.0254994],
    [152.784644, -23.3156906],
    [153.478380, -24.5738101],
])

# This line starts inside the dataset, exits to the north east in a big spike,
# then comes back in bounds again.
LINE_IN_OUT = shapely.LineString([
    [150.172296, -19.1132187],
    [150.985268, -20.7740421],
    [157.228887, -19.1951146],
    [152.784644, -23.3156906],
    [153.478380, -24.5738101],
])

# This line is completely outside the dataset and never intersects.
LINE_OUT_OF_BOUNDS = shapely.LineString([
    [154.414304, -15.7909029],
    [156.842378, -18.3706571],
    [158.533358, -21.2663872],
    [159.129537, -23.2525561],
])


@pytest.mark.tutorial
def test_transect():
    dataset = emsarray.tutorial.open_dataset('gbr4')
    transect = Transect(dataset, LINE_IN_OUT)

    # A whole bunch of segments
    assert len(transect.segments) > 0
    assert transect.intersection_bounds.shape == (len(transect.segments), 2)
    assert transect.linear_indexes.shape == (len(transect.segments),)

    # The path exits and re-enters the dataset once, so one hole
    assert transect.holes.shape == (1,)

    assert len(transect.points) == len(LINE_IN_OUT.coords)

    # Extract temp data, which
    temp_data = transect.extract('temp')
    assert temp_data.dims == ('time', 'k', 'index')
    assert temp_data.sizes['index'] == len(transect.segments)

    eta_data = transect.extract('eta')
    assert eta_data.dims == ('time', 'index')
    assert eta_data.sizes['index'] == len(transect.segments)

    botz_data = transect.extract('botz')
    assert botz_data.dims == ('index',)
    assert botz_data.sizes['index'] == len(transect.segments)


@pytest.mark.matplotlib
@pytest.mark.tutorial
def test_transect_make_artist_cross_section():
    dataset = emsarray.tutorial.open_dataset('gbr4')
    dataset = estimate_bounds_1d(dataset, 'zc')
    transect = Transect(dataset, LINE_INBOUNDS)

    figure = Figure(figsize=(12, 3))
    axes = figure.add_subplot()
    artist = transect.make_artist(axes, dataset['temp'].isel(time=0))

    assert isinstance(artist, CrossSectionArtist)
    assert artist in axes._children


@pytest.mark.matplotlib
@pytest.mark.tutorial
def test_transect_make_artist_transect():
    dataset = emsarray.tutorial.open_dataset('gbr4')
    transect = Transect(dataset, LINE_INBOUNDS)

    figure = Figure(figsize=(12, 3))
    axes = figure.add_subplot()
    artist = transect.make_artist(axes, dataset['eta'].isel(time=0))

    assert isinstance(artist, TransectStepArtist)
    assert artist in axes._children
