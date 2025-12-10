import cartopy.crs
import emsarray
import emsarray.plot
from cartopy.feature import GSHHSFeature
from matplotlib import pyplot

# Open the dataset
ds = emsarray.tutorial.open_dataset('austen')

# Make a figure
figure = pyplot.figure(figsize=(10, 8), layout='constrained')
axes = figure.add_subplot(projection=cartopy.crs.PlateCarree())
axes.set_aspect('equal', adjustable='datalim')

# Make an artist to plot eta
eta_artist = ds.ems.make_artist(
    axes, ds['eta'].isel(record=0),
    add_colorbar=False,
    clim=(-3, 3), edgecolor='face', cmap='Spectral_r',
)
figure.colorbar(eta_artist, ax=axes, location='right', label="metres")

# Draw some coastlines
coast = GSHHSFeature(scale='intermediate')
axes.add_feature(coast, facecolor='linen', edgecolor='darkgrey', linewidth=0.5)

# Finish setting up the plot
axes.autoscale()
axes.set_title("Sea surface height deviation")
axes.set_facecolor('aliceblue')

figure.savefig('plot-with-clim.png')

pyplot.show(block=True)
