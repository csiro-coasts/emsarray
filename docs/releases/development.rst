=============================
Next release (in development)
=============================

Improved support for multiple grids
===================================

In previous releases emsarray made the assumption that variables were primarily
defined on some grid of polygons.
This is true for rectilinear grids, curvilinear grids, spherical multiple cell grids,
and unstructured grids from `COMPAS <compas_>`_ models.
This is not true for the unstructured grids from `SCHISM <schism_>`_ models however
which define many variables on the vertices of the polygons, not the polygons themselves.
This assumption of polygons was limiting the datasets that emsarray could support.

With this update each grid kind is treated equally with no assumptions around face grids.
Each dataset can have multiple grid kinds,
and each of these grid kinds is represented by a :class:`.Grid` instance.
The set of grids for a dataset is available at :attr:`.Convention.grids`,
and the grid for a variable can be found using :meth:`.Convention.get_grid`.

The externally facing API is mostly backwards compatible
although many methods have been deprecated in favour of methods on the :class:`.Grid` class.
These methods will continue to be supported for this release at least.
`Convention.get_index_for_point` and `SpatialIndexItem` have been dropped instead of deprecated.
These methods were little used and backwards compatibility was complicated.
To make the transition smoother the assumption of a default polygon grid has been maintained,
however this default may be removed in future releases.

This change allows for plotting variables that are not defined on a polygon grid
all using the same API.
A new method :meth:`.Convention.make_artist` has been added
which will pick the correct matplotlib :class:`~matplotlib.artist.Artist` to plot a given variable.
Now :class:`~.conventions.ugrid.UGrid` datasets can plot variables defined on face grids and node grids
using the same API.
All artists returned by :meth:`~.Convention.make_artist`
have a :meth:`~.GridArtist.set_data_array` that can update the data drawn by an artist
which provides a consistent interface for making animations.

:ref:`More examples <examples>` have been added to the documentation to demonstrate this new functionality.

.. _ugrid: https://ugrid-conventions.github.io/ugrid-conventions/
.. _compas: https://research.csiro.au/cem/software/ems/hydro/unstructured-compas/
.. _schism: https://ccrm.vims.edu/schismweb/

Changelog
=========

* Reduce memory allocations when constructing polygons.
  This should allow opening even larger datasets
  (:pr:`200`, :pr:`207`).
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
* Use `PEP 695 <https://peps.python.org/pep-0695/>`_ style type parameters.
  This drops the `Index` and `GridKind` type variables
  which were exported in `emsarray.conventions`,
  which is a backwards incompatible change
  but is difficult to add meaningful backwards compatible support
  (:issue:`109`, :pr:`204`)
* Use default matplotlib colour map when plotting instead of "jet".
  In practice this will usually be "viridis"
  unless the user has changed their local defaults
  (:pr:`206`).
* Split the :mod:`emsarray.plot` module in to multiple files.
  It was getting unruly in size and was about to become larger again.
  The documentation has been updated to match the conceptual divisions.
  The new layout makes it easier to support matplotlib as an optional dependency
  (:pr:`208`).
* The handling of multiple grid kinds was rewritten from the ground up
  to properly support conventions like `unstructured grids <ugrid_>`_
  (:pr:`205`, also :issue:`189`, :issue:`175`, :issue:`121`, and :pr:`187`).
