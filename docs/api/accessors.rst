.. _accessor:

===============
xarray accessor
===============

Much of the functionality of :mod:`emsarray` is accessed
using the ``.ems`` attribute on an xarray :class:`xarray.Dataset`.

.. attribute:: dataset.ems
    :type: emsarray.formats.Format

    This accessor will attempt to determine the type of file format
    and create the appropriate :class:`~emsarray.formats.Format` instance for this dataclass
