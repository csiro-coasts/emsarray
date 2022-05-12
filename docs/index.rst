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

    dataset = emsarray.open_dataset('./tests/datasets/shoc_standard.nc')
    with open("geometry.geojson", "w") as f:
        json.dump(dataset.ems.make_geojson_geometry(), f)

Contents
========

.. toctree::
    :maxdepth: 1

    installing.rst
    concepts/index.rst
    cli.rst
    api/index.rst
    testing.rst

.. _xarray: https://xarray.pydata.org/
