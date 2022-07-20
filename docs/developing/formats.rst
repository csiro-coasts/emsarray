=============================
Supporting additional formats
=============================

:mod:`emsarray` allows developers to add support for additional formats
by creating a new subclass of the :class:`emsarray.formats.Format` class.
These additional formats can be either packaged as part of your application
or distributed as reusable and installable plugins.

Creating a subclass
===================

You've just made a new dataset format called
"Gigantic Revolutionary Awesome Spatial System" - GRASS for short.
You're making a Python package with a bunch of utilities
to help developers and scientists work with GRASS datasets,
called ``grass``.

To add support for GRASS to ``emsarray``, make a new Format subclass.
For this example, we will make a new module named ``grass.format``
with the following code:

.. code-block:: python

    import enum
    from typing import Tuple

    from emsarray.formats import Format, Specificity

A Format must specify an enum of the :ref:`different grids that it supports <grids>`.
If it only supports one grid, make an enum with a single member.

.. code-block:: python

    class GrassGridKind(enum.Enum):
        blade = 'blade'
        meadow = 'meadow'

A Format must specify the :ref:`format native index types <indexing>` it uses.
GRASS has two-dimensional indexes for both the blade and meadow grids,
making indexes like ``(kind, warp, weft)``:

.. code-block:: python

    GrassIndex = Tuple[GrassGridKind, int, int]

Then create a :class:`emsarray.formats.Format` subclass named ``Grass``.

.. code-block:: python

    class Grass(Format[GrassGridKind, GrassIndex]):

.. admonition:: TODO

   The rest of the class implementation:
   :meth:`.Format.check_dataset`,
   :meth:`.Format.ravel_index`,
   :meth:`.Format.unravel_index`,
   :meth:`.Format.get_grid_kind_and_size`,
   :meth:`.Format.make_linear`,
   :meth:`.Format.polygons`,
   :meth:`.Format.make_clip_mask`, and
   :meth:`.Format.apply_clip_mask`.


Registering as part of an application
=====================================

If you are making an application that needs to support GRASS,
or just experimenting with a new format type,
but don't intend on distributing the new format as a plugin,
you can use the :func:`~emsarray.formats.register_format` function.
This will add the format to the internal format registry.
It can be used as a decorator or called directly:

.. code-block:: python

    from emsarray.formats import Format, Specificity, register_format

    @register_format
    class Grass(...):

The format class will not be automatically discovered by ``emsarray``,
so you must ensure that the Python file containing the Grass subclass is imported
before you attempt to use it.
This can be done in your applications ``__init__.py`` as ``import grass.format``.

Distributing as a plugin
========================

``emsarray`` uses `entry points <https://packaging.python.org/en/latest/specifications/entry-points/>`_
to find format classes distributed as plugins.
Users can install your plugin and ``emsarray`` will automatically find the included subclass.

If you have created a Format subclass called ``Grass`` in the module ``grass.format``
include the following entry point in your ``setup.cfg``:

.. code-block:: ini

    [entry_points]
    emsarray.formats =
        Grass = grass.format:Grass

The ``name`` portion before the ``=`` is not used,
however we suggest using the same class name as your new format class.
The ``value`` portion after the ``=`` is the import path to your class,
then a ``:``, then the name of your class.
If your package contains multiple format classes, add one per line.

As a real world example, ``emsarray`` defines the following entry points:

.. literalinclude:: /../setup.cfg
   :prepend: [entry_points]
   :language: ini
   :start-at: emsarray.formats =
   :end-before: # emsarray.formats end
   :tab-width: 4
