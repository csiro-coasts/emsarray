.. _example-plot-with-clim:

===============================
Plot with specified data limits
===============================

The :meth:`Convention.make_poly_collection()` method passes all keyword arguments
on to :class:`~matplotlib.collections.PolyCollection`.
This can be used to customise the appearance of the collection
and to set parameters such as `clim`:

.. image:: /_static/images/plot-with-clim.png
   :alt: A plot of sea surface height deviation from the Australia wide AUSTEn dataset.
         The limits of the colour bar have been constrianed to (-3, 3).

Code
====

:download:`Download plot-with-clim.py example <plot-with-clim.py>`.

.. literalinclude:: plot-with-clim.py
   :language: python

