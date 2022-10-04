===================
emsarray.operations
===================

Operations are useful functions that
transform datasets, extract subsets, manipulate geometry, or other useful actions.
These functions do not depend on the specific format of the underlying dataset,
and behave the same across all supported formats.

.. toctree::
   :hidden:
   :glob:

   ./*

:mod:`emsarray.operations.depth`
    These operations manipulate datasets with a depth axis,
    such as the output of ocean models.

:mod:`emsarray.operations.geometry`
    These operations export the geometry of a dataset
    in a variety of different formats.

:mod:`emsarray.operations.triangulate`
    These operations triangulate the polygonal mesh of a dataset.
    This is useful in combination with visualisation packages such as
    Trimesh or Holoviews.
