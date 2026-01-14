"""
=======================
Vector plotting methods
=======================

To plot vector quantities such as currents we can use a quiver plot in matplotlib.
In the `GBR4` tutorial dataset
the u and v variables contain the x and y components of the vector
for each cell in the dataset.
Plotting every cell is straight forward for small areas:
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import xarray
from shapely import box

import emsarray
from emsarray import plot
from emsarray.operations import point_extraction

dataset_url = "https://thredds.nci.org.au/thredds/dodsC/fx3/gbr4_bgc_GBR4_H2p0_B3p1_Cfur_Dnrt/gbr4_bgc_simple_2024-01-16.nc"
dataset = emsarray.open_dataset(dataset_url, decode_timedelta=False)

surface_currents = dataset.ems.select_variables(["temp", "u", "v"]).isel(time=0, k=-1)

# Plot the points
figure = plt.figure(figsize=(8, 8), layout="constrained")
axes = figure.add_subplot(projection=dataset.ems.data_crs)
axes.set_title("Surface water temperature and currents near the Whitsundays")

temp = surface_currents["temp"]
temp.attrs['units'] = '°C'
temp_artist = dataset.ems.make_artist(axes, temp, cmap="coolwarm", clim=(27, 30), edgecolor="face")

current_artist = dataset.ems.make_artist(
    axes, (surface_currents["u"], surface_currents["v"]),
    scale=20, scale_units="width")

plot.add_coast(axes)
plot.add_gridlines(axes)

# Just show a small area over the Whitsundays
view_box = box(148.7, -20.4, 149.6, -19.8)
axes.set_extent(plot.bounds_to_extent(view_box.bounds))
axes.set_aspect("equal", adjustable="datalim")

# %%
# Plotting the entire dataset this way leads to the current vectors becoming a confusing mess however:

dataset = emsarray.tutorial.open_dataset('gbr4')

surface_currents = dataset.ems.select_variables(["temp", "u", "v"]).isel(time=0, k=-1)

# Plot the points
figure = plt.figure(figsize=(8, 10), layout="constrained")
axes = figure.add_subplot(projection=dataset.ems.data_crs)
axes.set_title("A bad plot of surface water temperature and currents over the entire reef")

temp = surface_currents["temp"]
temp.attrs['units'] = '°C'
temp_artist = dataset.ems.make_artist(axes, temp, cmap="coolwarm")

current_artist = dataset.ems.make_artist(
    axes, (surface_currents["u"], surface_currents["v"]),
    scale=20, scale_units="width")

plot.add_coast(axes)
plot.add_gridlines(axes)

# Show the entire model domain
axes.autoscale()
axes.set_aspect("equal", adjustable="datalim")

# %%
# For gridded datasets like this we can sample the current data at regular intervals to display only a subset of the vectors:

dataset = emsarray.tutorial.open_dataset('gbr4')

# Make an empty array of the same shape as the data, then select every nth cell in there
samples = xarray.DataArray(numpy.full(dataset.ems.grid_size[dataset.ems.default_grid_kind], False))
samples = dataset.ems.wind(samples)
samples[::10, ::10] = True
samples = dataset.ems.ravel(samples)

# Select the (x, y) coordinates and the (u, v) components of the sampled cells
surface_currents = dataset.ems.select_variables(["temp", "u", "v"]).isel(time=0, k=-1)
x, y = dataset.ems.face_centres[samples].T
u = dataset.ems.ravel(surface_currents["u"]).values[samples]
v = dataset.ems.ravel(surface_currents["v"]).values[samples]

# Plot the points
figure = plt.figure(figsize=(8, 10), layout="constrained")
axes = figure.add_subplot(projection=dataset.ems.data_crs)
axes.set_title("Surface water temperature and currents across the entire model domain")

temp = surface_currents["temp"]
temp.attrs['units'] = '°C'
temp_artist = dataset.ems.make_artist(axes, temp, cmap="coolwarm")

quiver = plt.quiver(
    x, y, u, v,
    scale=40, scale_units="width")
axes.add_collection(quiver)

plot.add_coast(axes)
plot.add_gridlines(axes)

# Show the entire model domain
axes.autoscale()
axes.set_aspect("equal", adjustable="datalim")

# %%
# Another approach is to plot vectors at regular points across the domain. This means that the vector locations are not tied to the grid geometry.

dataset = emsarray.tutorial.open_dataset('gbr4')

# Generate a mesh of points across the model domain
domain = box(*dataset.ems.bounds)
x = numpy.arange(domain.bounds[0], domain.bounds[2], 0.4)
y = numpy.arange(domain.bounds[1], domain.bounds[3], 0.4)
xx, yy = numpy.meshgrid(x, y)
points = pandas.DataFrame({
    'x': xx.flatten(),
    'y': yy.flatten(),
})

# Extract the surface current components at these locations
surface_currents = dataset.ems.select_variables(["temp", "u", "v"]).isel(time=0, k=-1)
surface_currents.load()
components = point_extraction.extract_dataframe(
    surface_currents, points, ('x', 'y'), missing_points='drop')

# Plot the points
figure = plt.figure(figsize=(8, 10), layout="constrained")
axes = figure.add_subplot(projection=dataset.ems.data_crs)
axes.set_title("Surface water temperature and currents across the entire model domain")

temp = surface_currents["temp"]
temp.attrs['units'] = '°C'
temp_artist = dataset.ems.make_artist(axes, temp, cmap="coolwarm")

quiver = plt.quiver(
    components['x'], components['y'], components['u'], components['v'],
    scale=30, scale_units="width")
axes.add_collection(quiver)

plot.add_coast(axes)
plot.add_gridlines(axes)

axes.autoscale()
axes.set_aspect("equal", adjustable="datalim")

