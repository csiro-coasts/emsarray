.. _cli:

======================
Command line interface
======================

:mod:`emsarray` provides a command line interface
allowing users to use some of the more common operations directly from their shell.

When :mod:`emsarray` is :ref:`installed <installing>`,
the command line interface will be available as ``emsarray``.
Run that command to get the full help pages:

.. code-block:: shell-session

    $ emsarray --help

Available commands
==================

``emsarray --version``
----------------------

Print the installed version of ``emsarray``.

``emsarray clip``
-----------------

Clip a dataset to a given GeoJSON geometry:

.. code-block:: shell-session

    $ emsarray clip "./input-file.nc" "./clip.geojson" "./output-file.nc"

Or clip a dataset to some geographic bounds:

.. code-block:: shell-session

    $ emsarray clip "./input-file.nc" "147.08,-43.67,147.30,-43.45" "./output-file.nc"

See ``emsarray clip --help`` for a full list of options.
