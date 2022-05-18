.. module:: emsarray

============================================
Coastal Environment Modelling dataset helper
============================================

The `emsarray` package provides a common interface
for working with the many model formats used at CSIRO.
It enhances `xarray`_ Datasets
and provides a set of common operations for manipulating datasets.

To use, open the dataset using either :func:`xarray.open_dataset` or :func:`emsarray.open_dataset`
and use the :ref:`dataset.ems <accessor>` attribute:

.. code-block:: python

    import emsarray
    import json

    dataset = emsarray.tutorial.open_dataset('gbr4')

    # Export dataset geometry as geojson
    with open("geometry.geojson", "w") as f:
        json.dump(dataset.ems.make_geojson_geometry(), f)

    # Plot the sea surface temperature for time = 0
    temp = dataset['temp'].isel(time=0, k=-1)
    dataset.ems.plot(temp)

.. image:: _static/images/gbr4_temp.png
   :alt: Plot of sea surface temperature from the GBR4 example file

Contents
========

.. toctree::
    :maxdepth: 1

    installing.rst
    concepts/index.rst
    examples/index.rst
    api/index.rst
    cli.rst
    testing.rst

.. _xarray: https://xarray.pydata.org/
