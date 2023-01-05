.. _accessor:

===============
xarray accessor
===============

Much of the functionality of ``emsarray`` is accessed
using the ``.ems`` attribute on an xarray :class:`~xarray.Dataset`.

.. attribute:: dataset.ems
    :type: emsarray.conventions.Convention

    This accessor will attempt to determine the geometry convention used
    and create the appropriate :class:`~emsarray.conventions.Convention`
    instance for this dataset
