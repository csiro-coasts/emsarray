============
emsarray.cli
============

.. module:: emsarray.cli.utils

If you want to write command line scripts that use ``emsarray``,
refer to the :ref:`tutorial <scripts>`.
This module provides a few helpers that make writing scripts much easier.

.. autofunction:: emsarray.cli.utils.console_entrypoint
.. autofunction:: emsarray.cli.utils.nice_console_errors
.. autofunction:: emsarray.cli.utils.add_verbosity_group
.. autofunction:: emsarray.cli.utils.set_verbosity


Command line arguments
----------------------

The following functions aid in converting command line flags
in to useful Python values.

.. autofunction:: emsarray.cli.utils.geometry_argument
.. autofunction:: emsarray.cli.utils.bounds_argument

Exceptions
----------

.. autoexception:: emsarray.cli.CommandException
