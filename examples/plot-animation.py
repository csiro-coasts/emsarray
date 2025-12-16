"""
.. _example-animated-plot:

=============
Animated plot
=============

Animated plots are possible with emsarray by using :class:`matplotlib.animation.FuncAnimation`
and calling :meth:`.GridArtist.set_data_array()` each frame.
A :class:`~matplotlib.animation.FuncAnimation` will call a function every frame where you can update the plot.
Each artist returned by :meth:`.Convention.make_artist()` has a :meth:`~.GridArtist.set_data_array()` method
which can be used to update the data in the plot.
In combination this makes animations in emsarray about as straight forward as making a static plot:
"""
import cartopy.crs
import datetime
import emsarray
import emsarray.plot
import numpy
from cartopy.feature import GSHHSFeature
from emsarray.utils import datetime_from_np_time
from matplotlib import pyplot
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Open the dataset
ds = emsarray.tutorial.open_dataset('kgari')

# Select just the surface water and get the current vectors
surface = ds.isel(k=-1)
u, v = surface['u'], surface['v']
# Compute the magnitude of the current vectors
magnitude = numpy.sqrt(u ** 2 + v ** 2)


# The dataset is in Australian Eastern Standard Time, UTC +10
aest_timezone = datetime.timezone(datetime.timedelta(hours=10))


# Make a figure
figure = pyplot.figure(figsize=(8, 8))
axes = figure.add_subplot(projection=cartopy.crs.PlateCarree())
axes.set_aspect('equal', adjustable='datalim')
coast = GSHHSFeature(scale='intermediate')
axes.add_feature(coast, facecolor='mistyrose', edgecolor='darkgrey', linewidth=0.5)
axes.set_facecolor('aliceblue')


# Make an artist to plot magnitude, selecting the first time step of data.
# When making an animation it is important to keep the artist in a variable
# so you can update the data frame by frame.
magnitude_artist = ds.ems.make_artist(
    axes, magnitude.isel(time=0),
    add_colorbar=False,
    clim=(0, 1), edgecolor='face', cmap='Oranges',
)
figure.colorbar(magnitude_artist, ax=axes, location='right', label="metres per second")

# Make an artist to plot the current vectors
uv_artist = ds.ems.make_artist(
    axes, (u.isel(time=0), v.isel(time=0)),
    scale=40)

def update_plot(frame: int) -> list[Artist]:
    # This function is called every frame and should update the plot with new data.

    # Disable the layout engine after the first frame, else the plot can shift slightly
    if frame > 0:
        figure.set_layout_engine('none')

    # Update the plot title to display the frame time
    frame_time = datetime_from_np_time(ds['time'].isel(time=frame).values)
    frame_time = frame_time.astimezone(aest_timezone)
    axes.set_title(f"Surface water currents\n{frame_time:%Y-%m-%d %H:%M %Z}")

    # Update the data being plotted by the artists
    magnitude_artist.set_data_array(magnitude.isel(time=frame))
    uv_artist.set_data_array((u.isel(time=frame), v.isel(time=frame)))

    # Return every artist that has been updated this frame
    return [axes.title, magnitude_artist, uv_artist]

# Render the first frame
update_plot(0)

# Finish setting up the plot
axes.autoscale()

animation = FuncAnimation(
    figure,  # The figure to animate
    update_plot,  # The function to call to update the plot data
    frames=ds.sizes['time'],  # How many frames of animation to render
)
