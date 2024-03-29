.. _indexing:

========
Indexing
========

Each geometry convention defines some number of :ref:`grids <grids>`.
Each location in that grid
- whether it is a face, edge, or node -
can be uniquely indexed.

Convention native indexing
==========================

As each geometry convention may define a different number of grids,
each convention has a different method of indexing data in these grids.
These are the *convention native indexes*.
Each :class:`~.conventions.Convention` implementation
has its own :data:`~.conventions._base.Index` type.

:mod:`CF grid datasets <.conventions.grid>` have only one grid - faces.
Each face can be indexed using two numbers *x* and *y*.
Indexes for CF grids are two-tuples of ``(y, x)``

:mod:`SHOC datasets <.conventions.shoc>` have four grids
- faces, left edges, back edges, and nodes.
Each of these grids can be indexed using two numbers *i* and *j*.
Indexes for SHOC datasets are three-tuples of ``(kind, j, i)``.

:mod:`UGRID datasets <.conventions.ugrid>` have three grids
- faces, edges, and nodes.
Each of these grids can be indexed using a single number.
Indexes for UGRID datasets are two-tuples of ``(kind, index)``.

Linear indexing
===============

Each grid can be flattened into a one dimensional array.
Indexes in to this flattened array are called *linear indexes*.
Linear indexes are used in many places.
There are methods to convert between a linear index and a native index.

Using indexes
=============

A convention native index can be transformed in to a linear index by calling
:meth:`Convention.ravel_index(native_index) <.Convention.ravel_index>`.
A linear index can be transformed in to a native index by calling
:meth:`Convention.wind_index(linear_index, grid_kind=grid_kind) <.Convention.wind_index>`.

To find the correct native index kind for a data variable,
call :meth:`.Convention.get_grid_kind(data_array) <.Convention.get_grid_kind>`.
This will return one of the :attr:`Convention.grid_kinds` members.
:attr:`.Convention.default_grid_kind` indicates which grid kind
represents the cells in this convention.
:attr:`.Convention.grid_size` indicates how large the index space is for each grid kind.

:attr:`.Convention.strtree`
is a :class:`spatial index <shapely.strtree.STRtree>`
of all cells in the dataset.
Querying it will return the linear index for any matching cells.

The cell polygons in :attr:`Convention.polygons <.Convention.polygons>`
are in linear index order.
If you have the linear index for a cell,
you can find its polygon by indexing directly in to this array.

:meth:`Convention.ravel` will transform the surface dimensions of a data variable
in to a linear one dimensional array.
The order of this flattened array matches the linear index order.
If you have the linear index for a cell,
you can find its value by indexing directly in to this array.
:meth:`Convention.wind` is the inverse operation.
It takes a linear data array and transforms it in to the same shape as the convention.
This can be used to create arbitrary new data arrays for a dataset
in a way completely agnostic of the underlying convention.
