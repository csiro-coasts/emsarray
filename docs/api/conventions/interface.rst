.. _interface:

====================
Convention interface
====================

.. contents::
   :local:

Each supported convention implements
the :class:`~emsarray.conventions.Convention` interface.

.. autofunction:: emsarray.open_dataset
.. autofunction:: emsarray.get_dataset_convention

.. autoclass:: emsarray.conventions.Convention
   :members:

.. autoclass:: emsarray.conventions.DimensionConvention

   .. automethod:: grid_dimensions
   .. automethod:: unpack_index
   .. automethod:: pack_index

.. autoclass:: emsarray.conventions.SpatialIndexItem
   :members:

.. autodata:: emsarray.conventions._base.GridKind
.. autodata:: emsarray.conventions._base.Index

.. autoclass:: emsarray.conventions.Specificity
   :members:
   :undoc-members:
