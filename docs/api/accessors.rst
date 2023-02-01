.. _accessor:

===============
xarray accessor
===============

Much of the functionality of ``emsarray`` is accessed
using the ``.ems`` attribute on an xarray :class:`~xarray.Dataset`.

.. attribute:: dataset.ems
    :type: emsarray.conventions.Convention

    This accessor will automatically determine
    the geometry convention used in the dataset
    and create the appropriate :class:`~emsarray.conventions.Convention`
    instance for this dataset.


Using the accessor
==================

Every xarray dataset will have a ``.ems`` attribute.
This includes datasets opened with
:func:`xarray.open_dataset`, :func:`xarray.open_mfdataset`,
and :func:`emsarray.open_dataset`:

.. code-block:: python

   import emsarray

   dataset = emsarray.tutorial.open_dataset('austen')
   dataset.ems.plot(dataset['eta'].isel(record=0))

Refer to the :class:`~emsarray.conventions.Convention` documentation
for a full list of available methods,
and the :doc:`available operations modules </api/operations/index>` for more.

.. _registering_accessor:

Registering the accessor
========================

emsarray registers the ``dataset.ems``
:func:`dataset accessor <xarray.register_dataset_accessor>`
when it is is first imported.

You can not access the ``dataset.ems`` attribute before emsarray is imported.
The following example will fail as emsarray has not been imported:

.. code-block:: python
   :caption: Incorrect

   >>> import xarray
   >>> dataset = xarray.tutorial.open_dataset('air_temperature')
   >>> dataset.ems.plot(dataset['air'].isel(time=0))
   AttributeError: 'Dataset' object has no attribute 'ems'

To fix this, make sure to import emsarray before opening your datasets:

.. code-block:: python
   :caption: Correct

   >>> import xarray
   >>> import emsarray
   >>> dataset = xarray.tutorial.open_dataset('air_temperature')
   >>> dataset.ems.plot(dataset['air'].isel(time=0))

The :func:`emsarray.open_dataset` shortcut is available.
Using this function ensures that the dataset accessor is registered,
and that the dataset uses a geometry convention that emsarray understands.
Using this function is not required to access emsarray functionality.

.. code-block:: python
   :caption: Using emsarray.open_dataset()

   >>> import emsarray
   >>> dataset = emsarray.open_dataset('path/to/dataset.nc')
   >>> dataset.ems.plot(dataset['temp'].isel(record=0))
