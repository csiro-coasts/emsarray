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

.. autoclass:: emsarray.conventions.Convention
   :members:

.. autoclass:: emsarray.conventions.DimensionConvention

   .. autoattribute:: grid_dimensions

.. autoclass:: emsarray.conventions.Grid
   :members:

.. autoclass:: emsarray.conventions.DimensionGrid
   :members: dimensions

.. type:: GridKind

    Some type that can enumerate the different :ref:`grid types <grids>`
    present in a dataset.
    This can be an :class:`enum.Enum` listing each different kind of grid.

    :type:`Index` values will be included in the feature properties
    of exported geometry from :mod:`emsarray.operations.geometry`.
    If the index type includes the grid kind,
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

.. autoclass:: emsarray.conventions.Specificity
   :members:
   :undoc-members:
