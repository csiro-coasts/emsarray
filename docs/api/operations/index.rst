===================
emsarray.operations
===================

Operations are useful functions that
transform datasets, extract subsets, manipulate geometry, or other useful actions.
These functions do not depend on the specific convention of the underlying dataset,
and behave the same across all supported conventions.

.. toctree::
   :hidden:
   :glob:

   ./*

:doc:`depth`
    These operations manipulate datasets with a depth axis,
    such as the output of ocean models.

:doc:`geometry`
    These operations export the geometry of a dataset
    in a variety of different formats.

:doc:`point_extraction`
    Functions to extract data from a dataset at a collection of points.

:doc:`triangulate`
    These operations triangulate the polygonal mesh of a dataset.
    This is useful in combination with visualisation packages such as
    Trimesh or Holoviews.
