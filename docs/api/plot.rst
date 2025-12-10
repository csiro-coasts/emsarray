=============
emsarray.plot
=============

.. automodule:: emsarray.plot

The :ref:`examples <examples>` section contains many worked examples on how to generate plots.
:ref:`example-plot-with-clim` is a good place to start.

Shortcuts
=========

These functions will generate an entire plot,
but have limited customisation options.

.. autofunction:: plot_on_figure
.. autofunction:: animate_on_figure

Utility methods
===============

These functions are useful shortcuts for getting something basic done.

.. autofunction:: add_coast
.. autofunction:: add_gridlines
.. autofunction:: add_landmarks
.. autofunction:: bounds_to_extent
.. autofunction:: polygons_to_collection
.. autofunction:: make_plot_title

Artist functions
================

These methods will create an artist for a particular kind of grid and variable.

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
