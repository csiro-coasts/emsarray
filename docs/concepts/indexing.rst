.. _indexing:

========
Indexing
========

Each format defines some number of :ref:`grids <grids>`.
Each location in that grid
- whether it is a face, edge, or node -
can be uniquely indexed.

Format native indexing
======================

As each format may define a different number of grids,
each format has a different method of indexing data in these grids.
These are the *format native indexes*.
Each :class:`~.formats.Format` implementation
has its own :data:`~.formats._base.Index` type.

:mod:`CF grid datasets <.formats.grid>` have only one grid - faces.
Each face can be indexed using two numbers *x* and *y*.
Indexes for CF grids are two-tuples of ``(y, x)``

:mod:`SHOC datasets <.formats.shoc>` have four grids
- faces, left edges, back edges, and nodes.
Each of these grids can be indexed using two numbers *i* and *j*.
Indexes for SHOC datasets are three-tuples of ``(kind, j, i)``.

:mod:`UGRID datasets <.formats.ugrid>` have three grids
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

A format native index can be transformed in to a linear index by calling
:meth:`Format.ravel_index(native_index) <.Format.ravel_index>`.
A linear index can be transformed in to a native index by calling
:meth:`Format.unravel_index(linear_index, kind) <.Format.unravel_index>`.

To find the correct native index kind for a data variable,
call :meth:`.Format.get_grid_kind_and_size`.
This will return a tuple of ``(kind, size)``.
``kind`` is the native index kind (face, edge, node, etc),
while ``size`` is the length of the linear index space for that grid.

Looking up a location in the :attr:`Format.spatial_index`
will return a :class:`.SpatialIndexItem` instance.
These values have a :attr:`~.SpatialIndexItem.linear_index` attribute
and a :attr:`~.SpatialIndexItem.index` attribute.

The cell polygons in :attr:`Format.polygons <.Format.polygons>`
are in linear index order.
If you have the linear index for a cell,
you can find its polygon by indexing directly in to this array.

:meth:`Format.make_linear` will the surface dimensions of a data variable
in to a linear one dimensional array.
The order of this flattened array matches the linear index order.
If you have the linear index for a cell,
you can find its value by indexing directly in to this array.
