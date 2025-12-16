"""
.. _example-kgari-transect:

====================
K'gari transect plot
====================
"""

import shapely
from matplotlib import pyplot

import emsarray
from emsarray import plot, transect

dataset_url = 'https://thredds.nci.org.au/thredds/dodsC/fx3/gbr4_H4p0_ABARRAr2_OBRAN2020_FG2Gv3_Dhnd/gbr4_simple_2022-10-31.nc'
dataset = emsarray.open_dataset(dataset_url).isel(time=-1)
dataset = dataset.ems.select_variables(['botz', 'temp'])

# %%
# The following is a :mod:`transect <emsarray.transect>` path
# starting in the Great Sandy Strait near K'gari,
# heading roughly North out to deeper waters:
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
landmarks = [
    ('Round Island', shapely.Point(152.9262543, -25.2878719)),
    ('Lady Elliot Island', shapely.Point(152.7145958, -24.1129146)),
]

# %%
# Plot a transect showing temperature along this path.

figure = transect.plot(
    dataset, line, dataset['temp'],
    figsize=(7.9, 3),
    bathymetry=dataset['botz'],
    landmarks=landmarks,
    title="Temperature",
    cmap='Oranges_r')
pyplot.show()

# %%
# The path of the transect can be plotted using matplotlib.

# Plot the path of the transect
figure = pyplot.figure(figsize=(5, 5), dpi=100)
axes = figure.add_subplot(projection=dataset.ems.data_crs)
axes.set_aspect(aspect='equal', adjustable='datalim')
axes.set_title('Transect path')
dataset.ems.make_artist(
    axes, 'botz', cmap='Blues', clim=(0, 2000), edgecolor='face',
    linewidth=0.5, zorder=0)
axes = figure.axes[0]
axes.set_extent(plot.bounds_to_extent(line.envelope.buffer(0.2).bounds))
axes.plot(*line.coords.xy, zorder=2, c='orange', linewidth=4)

plot.add_coast(axes, zorder=1)
plot.add_gridlines(axes)
plot.add_landmarks(axes, landmarks)

pyplot.show()
