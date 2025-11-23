=============================
Next release (in development)
=============================

* Reduce memory allocations when constructing polygons.
  This should allow opening even larger datasets
  (:pr:`200`).
* Add support for Python 3.14 and drop support for Python 3.11,
  following `SPEC-0 <https://scientific-python.org/specs/spec-0000/>`_.
  (:pr:`201`).
* Drop all previously deprecated functionality:
  support for shapely versions older than 2.0,
  support for cartopy versions older than 0.23.0,
  the ``emsarray.Format`` and ``emsarray.get_file_format()`` aliases,
  the renamed Convention methods ``_get_data_array()``,
  ``get_time_name()``, ``get_depth_name()``, ``get_all_depth_names()``,
  ``unravel_index()``, ``make_linear()``, ``make_patch_collection()``,
  ``spatial_index()``, ``get_grid_kind_and_size()``,
  and ``NonIntersectingPoints.indices``
  (:pr:`202`).
