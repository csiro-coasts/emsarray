.. _example-animated-plot:

=============
Animated plot
=============

Animated plots are possible with emsarray by using :class:`matplotlib.animation.FuncAnimation`
and calling :meth:`.GridArtist.set_data_array()` each frame.
A :class:`~matplotlib.animation.FuncAnimation` will call a function every frame where you can update the plot.
Each artist returned by :meth:`.Convention.make_artist()` has a :meth:`~.GridArtist.set_data_array()` method
which can be used to update the data in the plot.
In combination this makes animations in emsarray about as straight forward as making a static plot:

.. video:: /_static/images/animated-plot.mp4
   :poster: /_static/images/animated-plot.png
   :alt: A video of surface water currents around K'gari.
   :loop:
   :muted:
   :width: 100%

Code
====

Saving a video to a file requires `ffmpeg <https://www.ffmpeg.org/>`_
which can be installed using conda:

.. code-block:: shell

    $ conda install ffmpeg

:download:`Download animated-plot.py example <animated-plot.py>`.

.. literalinclude:: animated-plot.py
   :language: python

