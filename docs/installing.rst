.. _installing:

==========
Installing
==========

The easiest way of installing ``emsarray`` is using
`Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_.
``emsarray`` is published on the ``conda-forge`` channel.
You can install it in your Conda environment with:

.. code-block:: shell-session

   $ conda install -c conda-forge emsarray

This will install ``emsarray`` with all its :ref:`optional dependencies <extras>`.
Alternately you install only the core dependencies
by installing the ``emsarray-core`` package instead.

If you prefer, ``emsarray`` can be installed using ``pip``.

.. code-block:: shell-session

   $ pip install emsarray

Before installing ``emsarray`` via ``pip``
you have to ensure that the :ref:`non-Python dependencies are met <dependencies>`.
There are some optional dependencies for ``emsarray``.
You can install these by :ref:`choosing some extras at install time <extras>`.

.. _dependencies:

Dependencies
============

``emsarray`` depends on
`cartopy <https://scitools.org.uk/cartopy/docs/latest/installing.html>`_
for plotting.
This depends on the non-Python ``geos`` library.

These can be installed via your package manager or via ``conda``.
Installing from ``conda`` is the recommended approach
as these packages are often more up-to-date than the system packages
and it guarantees that compatible versions of ``geos`` is installed.

.. code-block:: shell-session

   $ conda create -n my-env
   $ conda activate my-env
   $ conda install cartopy

If ``geos`` is installed using your system package manager,
and ``cartopy`` is installed via pip,
you must ensure that you install versions of ``cartopy``
that are compatible with ``geos``.
``pip`` will not check for these version constraints for you.
A version mismatch between the Python and non-Python libraries
can lead to the installation failing,
or Python crashing when calling ``cartopy`` functions.

Building
========

On any computer, run the following commands from the root of the ``emsarray`` source directory to build a package:

.. code-block:: shell-session

    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip3 install --upgrade pip build
    $ rm -rf dist/
    $ python3 -m build

Two new files will be created in the ``dist/`` directory.
This is the Python package you can install in other environments.
Use either one of them when installing ``emsarray`` in your chosen environment:

.. code-block:: shell-session

    $ cd /path/to/other-project
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip3 install /path/to/emsarray/dist/emsarray-*-py3-none-any.whl

.. _extras:

Extras
======

When installed via ``pip``, ``emsarray`` can be installed with "extras".
These extra packages are optional.

For conda installs,
the ``emsarray`` package contains all the extras
and is equivalent to ``emsarray[complete]``.
``emsarray-core`` is equivalent to ``emsarray`` without any extras.

``plot``
--------

.. code-block:: shell

   $ pip install emsarray[plot]

Allows ``emsarray`` to produce plots, using :meth:`.Format.plot`.

``tutorial``
------------

.. code-block:: shell

   $ pip install emsarray[tutorial]

Installs packages required to access the tutorial datasets,
accessible via the :func:`emsarray.tutorial.open_dataset` method.

``complete``
------------

.. code-block:: shell

   $ pip install emsarray[complete]

Includes all extras.
Use this for the complete ``emsarray`` experience.

``testing``
-----------

The ``testing`` extras are intended for development.
When setting up a development environment for ``emsarray``,
clone the repository and install ``emsarray`` in editable mode
with the ``testing`` extras:

.. code-block:: shell

   $ pip install -e .[testing]
   $ pytest  # Run the test suite
   $ make -C docs html  # Build the docs
