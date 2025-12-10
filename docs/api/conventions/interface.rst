.. _interface:

====================
Convention interface
====================

.. contents::
   :local:

.. currentmodule:: emsarray.conventions

Each supported convention implements
the :class:`~emsarray.conventions.Convention` interface.

.. autofunction:: emsarray.open_dataset
.. autofunction:: emsarray.get_dataset_convention

Convention
==========

All dataset conventions have the following methods:

.. autoclass:: emsarray.conventions.Convention
   :members:

.. autoclass:: emsarray.conventions.Grid
   :members:

Concepts
========

.. type:: GridKind

    All datasets define variables on one or more  :ref:`grid<grids>`.
    :type:`GridKind` enumerates all the available grids for a specific convention,
    while :class:`Grid` provides methods for introspecting a particular grid.

    The :type:`GridKind` for a dataset is usually an :class:`enum.Enum` listing each different kind of grid.

    .. rubric:: Notes

    :type:`Index` values will be included in the feature properties
    of exported geometry from :mod:`emsarray.operations.geometry`.
    As the native index for a convention usually includes the grid kind,
    the grid kind needs to be JSON serializable.
    The easiest way to achieve this is to make your GridKind type subclass :class:`str`:

    .. code-block:: python

        class MyGridKind(str, enum.Enum):
            face = 'face'
            edge = 'edge'
            node = 'node'

    For cases where the convention only supports a single grid,
    a singleton enum can be used.

    More esoteric cases involving datasets with a potentially unbounded numbers of grids
    can use a type that supports this instead.

.. type:: Index

    An :ref:`index <indexing>` to a specific point on a grid in this convention.
    For conventions with :ref:`multiple grids <grids>` (e.g. cells, edges, and nodes),
    this should be a tuple whos first element is :type:`.GridKind`.
    For conventions with a single grid, :type:`.GridKind` is not required.

Dimension conventions
=====================

Most dataset conventions have grids that are uniquely identifiable by their dimensions.
For these conventions the DimensionConvention subclass provides many default implementations.
These details are most relevant for developers implementing new Convention subclasses.

.. autoclass:: emsarray.conventions.DimensionConvention

   .. autoattribute:: grid_dimensions

.. autoclass:: emsarray.conventions.DimensionGrid
   :members: dimensions

.. autoclass:: emsarray.conventions.Specificity
   :members:
   :undoc-members:
