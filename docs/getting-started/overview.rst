========
Overview
========

emsarray is an extension of :mod:`xarray`
that understands many geometry conventions used to represent data.
emsarray presents a unified interface to extract data at a point,
clip a dataset to a region,
or plot data to a figure.
It does this through the :doc:`Convention interface </api/conventions/index>`.

Most emsarray functionality is available through the
:doc:`dataset.ems </api/accessors>` attribute on any xarray dataset.
More utilities are provided in the
:doc:`emsarray.operations modules </api/operations/index>`.

Opening datasets
================

The simplest way to open a dataset is using :func:`emsarray.open_dataset`:

.. code-block:: python

   import emsarray
   dataset = emsarray.open_dataset(...)

A number of example datasets can be opened
using :func:`emsarray.tutorial.open_dataset`:

.. code-block:: python

   import emsarray
   dataset = emsarray.tutorial.open_dataset('gbr4')

As long as emsarray has been imported,
every xarray dataset will have the ``dataset.ems`` attribute.
This includes datasets opened using :func:`xarray.open_mfdataset`.
See :ref:`registering_accessor` for further information.

Extracting data from a point
============================

To get data at a single point
use :meth:`dataset.ems.select_point <.Convention.select_point>`:

.. code-block:: python

   from shapely.geometry import Point

   capricorn_group = Point(151.869, -23.386)
   point_data = dataset.ems.select_point(capricorn_group)

Plotting data
=============

Plots of a variable can be easily generated
using :meth:`dataset.ems.plot <.Convention.plot>`.
Pass the variable to plot to this method.
If the variable has time, depth, or other 'extra' dimensions
use :meth:`.DataArray.sel()` or :meth:`.DataArray.isel()` to select just one index:

.. code-block:: python

   # temp has both time and depth (k) dimensions.
   # Select just one index from these
   surface_temp = dataset['temp'].isel(time=0, k=-1)
   dataset.ems.plot(surface_temp)

.. image:: /_static/images/gbr4_temp.png
   :alt: Plot of sea surface temperature from the GBR4 example file

Exporting geometry
==================

:mod:`emsarray.operations.geometry`
can export dataset geometry to a number of formats.
This is useful if you want to examine the geometry QGIS or ArcGIS,
or use the geometry in another process.

.. code-block:: python

   from emsarray.operations import geometry

   geometry.write_shapefile(dataset, 'geometry.shp')
