=============================
Next release (in development)
=============================

* Add `missing_points` parameter
  to :func:`emsarray.operations.point_extraction.extract_points`
  and :func:`emsarray.operations.point_extraction.extract_dataframe`.
  Callers can now choose whether missing points raise an exception,
  are dropped from the returned dataset,
  or filled with a sensible fill value
  (:pr:`90`).
