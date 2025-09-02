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
* Fixed an issue with UGrid datasets when some of the mesh topology variables
  are not present in Dataset.data_vars as they are detected as coordinate variables
  (:issue:`159`, :pr:`188`).
* Load data frame by frame when plotting a transect animation.
  This has a backwards incompatible effect of not examining the whole data array to generate a reasonable *clim*.
  Only the first frame is used to generate the clim to avoid loading more data than required.
  The new clim parameter allows users to specify the data limits explicitly if this is insufficient
  (:pr:`191`).
