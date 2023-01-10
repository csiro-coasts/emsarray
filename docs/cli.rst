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

.. _emsarray clip:

``emsarray clip``
-----------------

Clip a dataset to a given GeoJSON geometry:

.. code-block:: shell-session

    $ emsarray clip "./input-file.nc" "./clip.geojson" "./output-file.nc"

Or clip a dataset to some geographic bounds:

.. code-block:: shell-session

    $ emsarray clip "./input-file.nc" "147.08,-43.67,147.30,-43.45" "./output-file.nc"

See ``emsarray clip --help`` for a full list of options.

.. _emsarray export-geometry:

``emsarray export-geometry``
----------------------------

Exports the geometry of a dataset to various formats.

.. code-block:: shell-session

    $ emsarray export-geometry gbr4.nc gbr.shp

The output format is guessed based on the extension of the output file,
or specified manually using the ``--format`` flag.

.. table::
   :align: left
   :width: 100%

   ================= ============= ==========
   Format            Flag          Extensions
   ================= ============= ==========
   GeoJSON           ``geojson``   ``.geojson``, ``.json``
   Shapefile         ``shapefile`` ``.shp``
   Well Known Text   ``wkt``       ``.wkt``
   Well Known Binary ``wkb``       ``.wkb``
   ================= ============= ==========

.. _emsarray extract-points:

``emsarray extract-points``
---------------------------

Extract the data at some points given in a CSV file:

.. code-block:: shell-session

    $ emsarray extract-points gbr4.nc gbr4-points.csv gbr4-points.nc

See ``emsarray extract-points --help`` for a full list of options.
