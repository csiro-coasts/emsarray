"""
============================
Multiple plots in one figure
============================
"""

from matplotlib import pyplot
import emsarray
from emsarray.utils import datetime_from_np_time
from emsarray.plot import add_coast

ds = emsarray.tutorial.open_dataset('austen')
upper_frame = ds.isel(record=0)
lower_frame = ds.isel(record=23)

figure = pyplot.figure(figsize=(7, 10), layout='constrained')
figure.suptitle("Sea surface height in the Timor Sea")
upper_axes, lower_axes = figure.subplots(2, 1, subplot_kw=dict(projection=ds.ems.data_crs))

upper_artist = ds.ems.make_artist(
    upper_axes, upper_frame['eta'], clim=(-3, 3), cmap='BrBG', add_colorbar=False)
lower_artist = ds.ems.make_artist(
    lower_axes, lower_frame['eta'], clim=(-3, 3), cmap='BrBG', add_colorbar=False)

# This roughly encompases the Timor Sea
extent = (124.38, 135.15, -17.0, -8.0)

add_coast(upper_axes)
upper_time = datetime_from_np_time(upper_frame['t'].values)
upper_axes.set_title(upper_time.strftime("%Y-%m-%d %H:%M +10:00"))
upper_axes.set_extent(extent)

add_coast(lower_axes)
lower_time = datetime_from_np_time(lower_frame['t'].values)
lower_axes.set_title(lower_time.strftime("%Y-%m-%d %H:%M +10:00"))
lower_axes.set_extent(extent)

figure.colorbar(
    lower_artist, ax=[upper_axes, lower_axes],
    location='right', fraction=0.05, label='meters')

pyplot.show()
