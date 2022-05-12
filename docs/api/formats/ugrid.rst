======================
emsarray.formats.ugrid
======================

.. automodule:: emsarray.formats.ugrid

.. autoclass:: UGrid
    :members: topology, open_dataset

Indexing
========

UGRID datasets have three :ref:`grids <grids>`: *face*, *edge* and *node*.
:class:`UGridKind` represents this.
Each grid is :ref:`indexed <indexing>` by a single integer.
The format native index type is :data:`UGridIndex`.

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

Masking
=======

.. autofunction:: mask_from_face_indices
.. autofunction:: buffer_faces
