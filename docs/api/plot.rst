=============
emsarray.plot
=============

.. automodule:: emsarray.plot

.. autofunction:: plot_on_figure
.. autofunction:: animate_on_figure

Plot helpers
============

These functions are useful shortcuts for getting something basic done.

.. autofunction:: add_coast
.. autofunction:: add_gridlines
.. autofunction:: add_landmarks
.. autofunction:: bounds_to_extent
.. autofunction:: polygons_to_collection
.. autofunction:: make_plot_title

Artist functions
================

.. autofunction:: make_polygon_scalar_collection
.. autofunction:: make_polygon_vector_quiver
.. autofunction:: make_polygon_contour
.. autofunction:: make_node_scalar_artist

Artists
=======

.. autoclass:: GridArtist
   :members: set_data_array

.. autoclass:: PolygonScalarCollection
   :members: from_grid

.. autoclass:: PolygonVectorQuiver
   :members: from_grid

.. autoclass:: PolygonTriContourSet
   :members: from_grid

.. autoclass:: NodeTriMesh
   :members: from_grid
