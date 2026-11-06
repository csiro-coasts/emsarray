=============================
Next release (in development)
=============================

* :attr:`.Convention.time_coordinate` now picks the variable with the most indications
  that it is the time coordinate when a dataset has several datetime variables,
  such as `time` and `forecast_reference_time`.
  Previously the first datetime variable with time units was used (:issue:`223`).
