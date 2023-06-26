=============================
Next release (in development)
=============================

* Fix an issue with negative coordinates in :func:`~emsarray.cli.utils.bounds_argument` (:pr:`74`).
* Add a new ``emsarray plot`` subcommand to the ``emsarray`` command line interface (:pr:`76`).
* Use :class:`matplotlib.collections.PolyCollection`
  rather than :class:`~matplotlib.collections.PatchCollection`
  for significant speed improvements
  (:pr:`77`).
* Added :func:`emsarray.utils.timed_func` for easily logging some performance metrics (:pr:`79`).
