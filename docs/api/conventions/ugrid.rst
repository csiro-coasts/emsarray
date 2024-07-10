==========================
emsarray.conventions.ugrid
==========================

.. automodule:: emsarray.conventions.ugrid

.. autoclass:: UGrid
    :members: topology

Indexing
========

UGRID datasets have three :ref:`grids <grids>`: *face*, *edge* and *node*.
:class:`UGridKind` represents this.
Each grid is :ref:`indexed <indexing>` by a single integer.
The convention native index type is :data:`UGridIndex`.

.. autoclass:: UGridKind
   :members:
   :undoc-members:

.. autodata:: UGridIndex

Topology
========

.. autoclass:: Mesh2DTopology
    :members:

.. autoclass:: NoEdgeDimensionException
    :members:

.. autoclass:: NoConnectivityVariableException
    :members:

Masking
=======

.. autofunction:: mask_from_face_indexes
.. autofunction:: buffer_faces
