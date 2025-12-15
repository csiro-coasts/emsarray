=============
emsarray.plot
=============

.. currentmodule:: emsarray.plot

Plotting an entire figure
=========================

The :func:`plot_on_figure` and :func:`animate_on_figure` functions
will generate a simple plot of any supported variables.
These functions are intended as quick and simple ways of exploring a dataset
and have limited customisation options.
Consult the :ref:`examples gallery <examples>`
for demonstrations on making more customised plots.

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
