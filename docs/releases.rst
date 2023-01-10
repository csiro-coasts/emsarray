=============
Release notes
=============

0.4.0 (in development)
======================

* Add CF Convention attributes to coordinate variables when extracting points
  (:pr:`34`)
* Reworked how command line entrypoints are discovered (:pr:`35`)
* Added brief tutorial on writing command line Python scripts
  using ``emsarray`` (:pr:`35`)
* Various documentation fixes (:pr:`42`, :pr:`43`)
* Added Python 3.11 support to CI (:pr:`44`)
* Allow manual binding of conventions to datasets (:pr:`45`)
* Renamed ``Format`` to :class:`~emsarray.conventions.Convention` (:pr:`46`)
* Used :func:`shapely.polygons` when generating geometry.
  This results in a massive performance improvement.
  The minimum version of Shapely has been bumped to 2.0.0 (:pr:`47`)
* Add :ref:`emsarray export-geometry` command (:pr:`48`)
* Use coordinate bounds from the dataset when making polygons (:pr:`49`)

0.3.1
=====

* Fixed 'Release notes' URL in ``setup.cfg`` package metadata.

0.3.0
=====

* Use variable and dimension names as-is, instead of casting to strings.
  :mod:`xarray` internally treats all variable and dimension names as Hashable,
  without assuming they are strings.
  NetCDF4 files always use string names,
  so this change should not affect you if you only use NetCDF4 datasets
  (:pr:`25`).
* Fix error when UGRID `start_index` is the string `"0"` or `"1"`.
  The conventions imply this should be an integer type,
  however real datasets use a string value here so a tolerant implementation is useful
  (:pr:`26`, :pr:`csiro-coasts/emsarray-data#1`).
* Split :mod:`emsarray.operations` in to separate modules
  (:pr:`27`).
* Add :doc:`api/operations/geometry` module
  which can export dataset geometry to GeoJSON and Shapefiles
  (:pr:`28`).
* Add :meth:`.Convention.drop_geometry()` method
  (:pr:`29`).
* Add :doc:`api/operations/point_extraction` module
  and :ref:`emsarray extract-points` command line entry point
  which can extract point data from a dataset
  (:pr:`29`).

0.2.0
=====

* Added support for :doc:`additional conventions via plugins </developing/conventions>`
  (:pr:`11`).
* Added support for one-based indexing in UGRID datasets
  (:pr:`14`).
* Buffering around clip regions in :meth:`.Convention.make_clip_mask` is now optional
  (:issue:`12`, :pr:`20`).
* Removed dependency on SciPy, added missing dependency to conda package.

0.1.0
=====

* Initial public release
