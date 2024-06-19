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
  (`update notice <https://opus.nci.org.au/display/NDP/THREDDS+Upgrade>`,
  :issue:`135`, :pr:`136`, :pr:`csiro-coast/emsarray-data#2`).
