=============================
Next release (in development)
=============================

* Fixed issues with detecting topology variables in SHOC datasets
  where some variables are lacking a `standard_name` (:issue:`178`, :pr:`180`).
* Updated the tests to be compatible with the latest xarray versions (:pr:`182`).
* Added an example to the documentation showing
  :ref:`how to set the clim parameter in plots <example-plot-with-clim>`
  (:pr:`179`).
* Bumped pinned dependencies (:pr:`183`).
* Fixed :func:`emsarray.utils.datetime_from_np_time`
  when the system timezone is not UTC and a specific timezone is requested
  (:issue:`176`, :pr:`183`).
