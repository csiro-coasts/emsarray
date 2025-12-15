=========================
emsarray.conventions.grid
=========================

.. automodule:: emsarray.conventions.grid

.. autoclass:: CFGrid
   :members: __init__, topology

.. autoclass:: CFGrid1D

.. autoclass:: CFGrid2D

Indexing
========

CF grid datasets have one :ref:`grid <grids>`: *face*.
:class:`CFGridKind` represents this.
Each face is :ref:`indexed <indexing>` by two integers *x* and *y*.

.. autoclass:: CFGridKind
   :members:
   :undoc-members:

Topology
========

.. autoclass:: CFGridTopology
   :members:

.. autoclass:: CFGrid1DTopology

.. autoclass:: CFGrid2DTopology
