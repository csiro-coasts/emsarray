=============================
Next release (in development)
=============================

* Fix a ``FutureWarning`` on accessing :attr:`xarray.Dataset.dims`
  with xarray >= 2023.12.0
  (:pr:`124`, :pr:`pydata/xarray#8500`).
* Fix an error when creating a transect plot that does not intersect the model geometry.
  Previously this would raise a cryptic error, now it returns an empty transect dataset
  (:issue:`119`, :pr:`120`).
* Drop dependency on importlib_metadata.
  This was only required to support Python 3.8, which was dropped in a previous release
  (:issue:`122`, :pr:`125`).
* Fix an error with ``ShocSimple.get_all_depth_names()``
  when the dataset had no depth coordinates
  (:issue:`123`, :pr:`126`).
* Use `PEP 585 generic type annotations <https://peps.python.org/pep-0585/>`_
  and stop using `PEP 563 postponed annotation evaluation <https://peps.python.org/pep-0563/>`_
  (:issue:`109`, :pr:`127`).
* Use `pytest-mpl <https://pypi.org/project/pytest-mpl/>`_ for image comparison tests
  for testing plotting methods
  (:pr:`128`).
* Update all URLs to the NCI THREDDS server
  (`update notice <https://opus.nci.org.au/display/NDP/THREDDS+Upgrade>`_,
  :issue:`135`, :pr:`136`, :pr:`csiro-coast/emsarray-data#2`).
* Correct all references to K'gari, formerly Fraser Island
  (:issue:`133`, :pr:`csiro-coast/emsarray-data#2`, :pr:`134`).
* Bump minimum versions of dependencies, update pinned dependencies for CI.
  Officially support numpy version 2.0.0
  (:pr:`137`).
* Lint Python code in `docs/` and `scripts/`
  (:pr:`141`).
* Add :func:`emsarray.utils.name_to_data_array()` and :func:`~emsarray.utils.data_array_to_name()` functions.
  Allow more functions to interchangeably take either a data array or the name of a data array
  (:pr:`142`).
* Add :attr:`.Convention.depth_coordinates` and :meth:`.Convention.get_depth_coordinate_for_data_array()`. Deprecate functions :meth:`.Convention.get_depth_name()`, :meth:`.Convention.get_all_depth_names()`, and :meth:`Convention.get_time_name()`. Remove deprecated functions ``Convention.get_depths()`` and ``Convention.get_times()`` (:pr:`143`).
* Swap to using `pyproject.toml` for all project metadata (:pr:`145`).
