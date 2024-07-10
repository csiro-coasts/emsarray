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
GRASS grids use indexes with two coordinates for fields,
and one index for fences:

.. literalinclude:: ./grass.py
   :start-at: GrassIndex
   :lines: 1-4

Specifying the index type is only used for type checking,
it is not referred to or enforced at runtime.

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

:meth:`.DimensionConvention.unpack_index` and :meth:`.DimensionConvention.pack_index`
transform between native index types and a grid kind and indexes.
The native representation must be representable as JSON for GeoJSON export support.
The simplest representation is a tuple of (grid_kind, indexes):

.. literalinclude:: ./grass.py
   :pyobject: Grass.unpack_index

.. literalinclude:: ./grass.py
   :pyobject: Grass.pack_index

:meth:`.DimensionConvention.grid_dimensions` specifies which dataset dimensions
each grid kind is defined on.
This method can introspect the dataset to determine the correct dimensions if required.
This method should be cached.

.. literalinclude:: ./grass.py
   :pyobject: Grass.grid_dimensions

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
include the following entry point in your ``pyproject.toml``:

.. code-block:: ini

    [project.entry-points."emsarray.conventions"]
    Grass = "grass.convention:Grass"

The ``name`` portion before the ``=`` is not used,
however we suggest using the same class name as your new convention implementation.
The ``value`` portion after the ``=`` is the import path to your class,
then a ``:``, then the name of your class.
If your package contains multiple convention implementations, add one per line.

As a real world example, ``emsarray`` defines the following entry points:

.. literalinclude:: /../pyproject.toml
   :language: toml
   :start-at: [project.entry-points."emsarray.conventions"]
   :end-before: # emsarray.conventions end
   :tab-width: 4
