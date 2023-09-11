=============================
Next release (in development)
=============================

* Use 'ravel' and 'wind' as antonyms instead of 'ravel' and 'unravel'.
  English is weird. 'Ravel' and 'unravel' mean the same thing!.
  (:pr:`100`, :pr:`104`).
* Added new :class:`emsarray.conventions.DimensionConvention` subclass.
  For conventions with multiple grids defined on unique subsets of dimensions
  this base class will provide a number of default method implementations.
  All existing conventions have been updated to build off this base class.
  (:pr:`100`)
* Remove workaround for `pydata/xarray#6049 <https://github.com/pydata/xarray/pull/6049>`_ (:pr:`101`).
* Add :meth:`.Convention.wind()` method as the inverse to :meth:`.Convention.ravel()`
  (:pr:`102`, :pr:`104`).
* Add :attr:`.Convention.strtree()` spatial index,
  deprecate :attr:`.Convention.spatial_index()`.
  The old attribute was a compatibility shim around Shapely 1.8.x STRtree implementation.
  Now that the minimum version of Shapely is 2.0, the STRtree can be used directly.
  (:pr:`103`).
