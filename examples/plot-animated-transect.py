"""
=================
Animated transect
=================

Transect and cross section plots can be animated using
:meth:`.TransectArtist.set_data_array()` to update the data.
"""
import datetime

import shapely
import matplotlib.pyplot as plt
import pandas
import xarray
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, EngFormatter

import emsarray
from emsarray import transect, utils
from emsarray.operations import depth

dataset = emsarray.tutorial.open_dataset('kgari')

# Select only the variables we want to plot.
dataset = dataset.ems.select_variables(['eta'])

# %%
# The following is a :mod:`transect <emsarray.transect>` path
# starting in the Great Sandy Strait near K'gari,
# heading roughly North out to deeper waters:
north_transect = transect.Transect(dataset, shapely.LineString([
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
]))


# %%
# Now we set up a figure and add a cross section artist.

# sphinx_gallery_defer_figures
# sphinx_gallery_capture_repr_block = ()

figure = plt.figure(figsize=(7.8, 3), layout='constrained', dpi=100)
axes = figure.add_subplot()

axes.set_title('Sea surface height above geoid')

# Draw the sea surface height
eta = dataset['eta']
eta_artist = north_transect.make_artist(
    axes, eta.isel(time=0))

# Add a time label
aest = datetime.timezone(datetime.timedelta(hours=10))
datetime_labels = [
    utils.datetime_from_np_time(date).astimezone(aest).strftime('%Y-%m-%d %H:%M')
    for date in dataset['time'].values
]
datetime_annotation = axes.annotate(
    datetime_labels[0],
    xy=(5, 5), xycoords='axes points',
    verticalalignment='bottom', horizontalalignment='left')

# Set up the axes
transect.setup_distance_axis(
    north_transect, axes)
axes.set_ylim(-1.5, 1.5)
axes.set_ylabel("Height")
axes.yaxis.set_major_formatter('{x:1.1f} m')
axes.axhline(y=0, linestyle='--', color='grey', linewidth=0.5)

# %%
# Finally we set up the animation.
# The ``update()`` function is called every frame to update the plot with new data.
# The :meth:`.TransectArtist.set_data_array()` function does all the hard work here.

def update(frame: int) -> list[Artist]:
    eta_artist.set_data_array(eta.isel(time=frame))
    datetime_annotation.set_text(datetime_labels[frame])
    return [eta_artist, datetime_annotation]

animation = FuncAnimation(figure, update, frames=eta.sizes['time'])
