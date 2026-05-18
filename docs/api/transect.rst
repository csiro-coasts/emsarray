
=================
emsarray.transect
=================

.. module:: emsarray.transect

This module provides methods for extracting and plotting data
along transects through your datasets.
A transect path is represented as a :class:`shapely.LineString`.
Data along the transect can be extracted in to a new :class:`xarray.Dataset`,
or plotted using :meth:`Transect.make_artist`.

Currently it is only possible to take transects through grids with polygonal geometry.
Taking transects through other kinds of geometry is a planned future enhancement.

Examples
========

.. minigallery::

   ../examples/plot-kgari-transect.py
   ../examples/plot-animated-transect.py

Transects
=========

These classes find the intersection of a :class:`shapely.LineString` with a dataset
and provide methods to introspect this intersection, plot data along this path,
and extract data along this path.

.. autoclass:: Transect
   :members:

.. autoclass:: TransectPoint()
   :members:

.. autoclass:: TransectSegment()
   :members:

Artists
=======

These classes plot data along a transect.
Transect artists are normally created by calling :meth:`.Transect.make_artist`.

.. module:: emsarray.transect.artists

.. autoclass:: TransectArtist()
   :members: set_data_array

.. autoclass:: CrossSectionArtist()
   :members: from_transect

.. autoclass:: TransectStepArtist()
   :members: from_transect

Utilities
=========

.. currentmodule:: emsarray.transect

.. autofunction:: plot
.. autofunction:: setup_distance_axis
.. autofunction:: setup_depth_axis
