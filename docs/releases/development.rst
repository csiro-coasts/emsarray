=============================
Next release (in development)
=============================

* Add `sphinx-gallery <https://sphinx-gallery.github.io/>`_ to the docs
  to render the example gallery (:pr:`210`).
* Update `emsarray-data <https://github.com/csiro-coasts/emsarray-data>`_
  to version 1.0.0, which includes the
  `SCHISM-WWMIII hydrological and wave model hindcast for Vanuatu <https://data.csiro.au/collection/65060>`_ dataset
  (:pr:`212`).
* Add example showing
  :ref:`multiple ways of plotting vector data <sphx_glr_examples_plot-vector-methods.py>`
  (:pr:`213`, :pr:`215`).
* Add :attr:`.Grid.centroid_coordinates` attribute
  (:pr:`214`).
* Stop using pytz.
  The Python datetime module now has sufficient functionality,
  the external dependency is no longer required.
  pytz was included as a dependency of pandas,
  and pandas recently dropped pytz which broke things.
  (:pr:`219`).
* Defer ShocSimple coordinate detection to the CFGrid2D base class
  (:issue:`217`, :pr:`218`).
* Split `tests.utils` in to multiple `tests.helpers` submodules
  (:pr:`220`).
* Split `tests.test_utils` in to multiple `tests.utils.test_component` submodules
  (:pr:`220`).
