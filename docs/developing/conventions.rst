.. _additional_conventions:

=================================
Supporting additional conventions
=================================

:mod:`emsarray` allows developers to add support for additional geometry conventions
by creating a new subclass of the :class:`emsarray.conventions.Convention` class.
These additional conventions can be either packaged as part of your application
or distributed as reusable and installable plugins.

Creating a subclass
===================

You've just made a new dataset geometry convention called
"Gigantic Revolutionary Awesome Spatial System" - GRASS for short.
You're making a Python package with a bunch of utilities
to help developers and scientists work with GRASS datasets,
called ``grass``.

To add support for GRASS to ``emsarray``,
make a new :class:`~emsarray.conventions.Convention` subclass.
For this example, we will make a new module named ``grass.convention``.
The complete implementation of the :download:`Grass class is available <./grass.py>`.
The following is a guided walk through developing this class.

We will need the following imports:

.. literalinclude:: ./grass.py
   :language: python
   :start-after: # > imports
   :end-before: # <
   :tab-width: 4

Grids and indexes
-----------------

A Convention must specify an enum of the :ref:`different grids that it supports <grids>`.
If it only supports one grid, make an enum with a single member.

.. literalinclude:: ./grass.py
   :pyobject: GrassGridKind

A Convention must specify the :ref:`convention native index types <indexing>` it uses.
GRASS has two-dimensional indexes for both the blade and meadow grids,
making indexes like ``(kind, warp, weft)``:

.. literalinclude:: ./grass.py
   :start-at: GrassIndex
   :lines: 1

Convention class
----------------

Create a :class:`emsarray.conventions.Convention` subclass named ``Grass``,
and implement all the methods below.

.. literalinclude:: ./grass.py
   :start-at: class Grass(
   :end-at: default_grid_kind

:meth:`.Convention.check_dataset` introspects a :class:`xarray.Dataset`
and returns a value indicating whether this convention implementation can understand the dataset.

.. literalinclude:: ./grass.py
   :pyobject: Grass.check_dataset

The :meth:`.Convention.ravel_index` and :meth:`.Convention.unravel_index` methods
convert between linear and native indexes.
It is important that the functions are the inverse of one another ---
that is ``grass.unravel_index(grass.ravel_index(index)) == index``.

.. literalinclude:: ./grass.py
   :pyobject: Grass.ravel_index

.. literalinclude:: ./grass.py
   :pyobject: Grass.unravel_index

:meth:`.Convention.get_grid_kind_and_size` determines
what kind of grid a :class:`xarray.DataArray` contains.
This is usually done by examining the dimensions of the DataArray.

.. literalinclude:: ./grass.py
   :pyobject: Grass.get_grid_kind_and_size

:meth:`.Convention.make_linear` takes a :class:`~xarray.DataArray`
and flattens the grid dimensions.
The returned data array can be indexed using linear indexes.
Non-grid dimensions such as time and depth should be left as-is.
:func:`emsarray.utils.linearise_dimensions` is very useful here.

.. literalinclude:: ./grass.py
   :pyobject: Grass.make_linear

:meth:`.Convention.selector_for_index` takes a native index
and returns a dict that can be passed to :meth:`xarray.Dataset.isel`.
This selector can be used to subset the dataset to a single grid index.

.. literalinclude:: ./grass.py
   :pyobject: Grass.selector_for_index

:attr:`.Convention.polygons` is an array of :class:`shapely.Polygon` instances,
one for each face in the dataset.
If a cell does not have a valid polygon
--- for example, if the coordinates for that polygon have been dropped
or are outside of the valid region
--- that index must be ``None``.
It is strongly encouraged to use ``@cached_property`` for this property,
as it is typically slow to run.

.. literalinclude:: ./grass.py
   :pyobject: Grass.polygons

The last thing to implement is clipping datasets,
via the :meth:`.Convention.make_clip_mask`
and :meth:`.Convention.apply_clip_mask` methods.
Implementers are encouraged to look at existing Convention implementations
for concrete examples.

.. literalinclude:: ./grass.py
   :pyobject: Grass.make_clip_mask

.. literalinclude:: ./grass.py
   :pyobject: Grass.apply_clip_mask


Registering as part of an application
=====================================

If you are making an application that needs to support GRASS,
or just experimenting with a new convention type,
but don't intend on distributing the new convention implementation as a plugin,
you can use the :func:`~emsarray.conventions.register_convention` function.
This will add the convention to the internal convention registry.
It can be used as a decorator or called directly:

.. code-block:: python

    from emsarray.conventions import Convention, Specificity, register_convention

    @register_convention
    class Grass(...):

The convention implementation will not be automatically discovered by ``emsarray``,
so you must ensure that the Python file containing the Grass subclass is imported
before you attempt to use it.
This can be done in your applications ``__init__.py`` as ``import grass.convention``.

Distributing as a plugin
========================

``emsarray`` uses `entry points <https://packaging.python.org/en/latest/specifications/entry-points/>`_
to find convention implementations distributed as plugins.
Users can install your plugin and ``emsarray`` will automatically find the included subclass.

If you have created a convention subclass called ``Grass``
in the module ``grass.convention``
include the following entry point in your ``setup.cfg``:

.. code-block:: ini

    [entry_points]
    emsarray.conventions =
        Grass = grass.convention:Grass

The ``name`` portion before the ``=`` is not used,
however we suggest using the same class name as your new convention implementation.
The ``value`` portion after the ``=`` is the import path to your class,
then a ``:``, then the name of your class.
If your package contains multiple convention implementations, add one per line.

As a real world example, ``emsarray`` defines the following entry points:

.. literalinclude:: /../setup.cfg
   :prepend: [entry_points]
   :language: ini
   :start-at: emsarray.conventions =
   :end-before: # emsarray.conventions end
   :tab-width: 4
