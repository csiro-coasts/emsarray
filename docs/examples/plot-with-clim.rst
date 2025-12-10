.. _example-plot-with-clim:

===============================
Plot with specified data limits
===============================

:meth:`.Convention.make_artist()` method will create an appropriate
:class:`~matplotlib.artist.Artist` to plot a variable.
The artist can be customised by passing kwargs to :meth:`~.Convention.make_artist()`.
The :meth:`~.Convention.make_artist()` documentation for each convention
will describe what artists are created for what kinds of variables.

Typically a scalar variable defined on a polygon grid
will be drawn using a :class:`~matplotlib.collections.PolyCollection`.
This class accepts options such as `clim` and `cmap`
which can be used to customise the appearance of the plot.

.. image:: /_static/images/plot-with-clim.png
   :alt: A plot of sea surface height deviation from the Australia wide AUSTEn dataset.
         The limits of the colour bar have been constrianed to (-3, 3).

Code
====

:download:`Download plot-with-clim.py example <plot-with-clim.py>`.

.. literalinclude:: plot-with-clim.py
   :language: python

