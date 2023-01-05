.. _grids:

=====
Grids
=====

Each :class:`geometry convention <.Convention>` defines data across a geographic area.
That area is divided up in to cells or *faces*.
These faces define one grid.
Dataset variables can represent data on this grid,
which indicates some value at the centre of the face.
An example of this is temperature,
a scalar value defined at each face centre.

Some datasets define additional grids.
Water flowing from one face to its neighbour
is commonly represented as flux through a shared *edge*,
from one face to another.
These edges represent another grid.
Some conventions also define variables on face vertices, called *nodes*.
Nodes represent a third grid.
This is represented by the :data:`~.conventions._base.GridKind` type variable.

Each of the faces, edges, and nodes define an area, line, or point.
These areas, lines, or points exist at some geographic location.
In this sense they exist as horizontal or surface dimensions.

Each convention can represent data at multiple depth layers.
These depth layers add a third, vertical dimension to the cell, edge, or node grid.

Each convention can represent data at multiple time steps.
Time is a fourth dimension.

Examples
========

The ``botz`` variable represents the depth of the ocean floor for a cell.
This variable does not vary across time, and it does not use the depth dimension.
``botz`` is two dimensional.

The ``eta`` variable represents the sea surface height for a cell.
It will vary across time.
As this variable is defined at the surface, it will not use the depth axis.
It is three dimensional - two spatial dimensions and one time dimension.

The ``temp`` variable represents the temperature of the ocean for a cell.
It will vary across time and by depth.
It is four dimensional - three spatial dimensions and one time dimension.

Current flow between cells can be defined on edges between cells.
This variable is often called ``u1``.
Current flow will vary by time and by depth.
This variable is not defined on the same grid as ``temp``,
but on the edges between cells instead.
