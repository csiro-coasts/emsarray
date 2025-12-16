=============
emsarray.plot
=============

.. module:: emsarray.plot

Plotting an entire figure
=========================

The :func:`plot_on_figure` and :func:`animate_on_figure` functions
will generate a simple plot of any supported variables.
These functions are intended as quick and simple ways of exploring a dataset
and have limited customisation options.

The :ref:`examples <examples>` section contains many worked examples
on how to generate plots with more customisations.
:ref:`sphx_glr_examples_plot-with-clim.py` is a good place to start.

Shortcuts
=========

These functions will generate an entire plot,
but have limited customisation options.

.. autofunction:: plot_on_figure
.. autofunction:: animate_on_figure

Shortcuts
=========

These functions make common plotting operations simple.
They are designed as quick shortcuts
and aim for ease of use and simplicity over being fully featured.
Most of these functions defer to other parts of matplotlib or cartopy for the actual implementation,
and users are encouraged to call these underlying implementations directly if more customisation is required.

.. autofunction:: add_coast
.. autofunction:: add_gridlines
.. autofunction:: add_landmarks

Utilities
=========

.. autofunction:: polygons_to_collection
.. autofunction:: bounds_to_extent
.. autofunction:: make_plot_title

Artist functions
================

.. module:: emsarray.plot.artists

These functions will make a matplotlib :class:`~matplotlib.artist.Artist`
that can plot variables directly from a support emsarray dataset.
These functions and the associated artists can be imported from
:mod:`emsarray.plot.artists`.

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
