"""
SHOC (Sparse Hydrodynamic Ocean Code) is the hydrodynamic model used by the
`Coastal Environmental Modelling Team <https://research.csiro.au/cem/>`_
at `CSIRO <https://www.csiro.au/>`_.
There are two related conventions: SHOC standard and SHOC simple.

SHOC standard is an :mod:`Arakawa C grid <.arakawa_c>` dataset
with known coordinate names.
SHOC standard datasets can be simplified in to a SHOC simple dataset,
which is a :class:`curvilinear CF grid <.grid.CFGrid2D>`
with known coordinate names.

See Also
--------
`SHOC documentation <https://research.csiro.au/cem/software/ems/hydro/strucutured-shoc/>`_
"""
from __future__ import annotations

import logging
from functools import cached_property
from typing import Hashable, List, Optional, Tuple

import xarray

from emsarray.exceptions import NoSuchCoordinateError

from ._base import Specificity
from .arakawa_c import ArakawaC, ArakawaCGridKind
from .grid import CFGrid2D, CFGrid2DTopology

logger = logging.getLogger(__name__)


class ShocStandard(ArakawaC):
    """
    SHOC datasets are :class:`.ArakawaC` datasets
    with predefined coordinate names for the four grids.
    """

    #: Coordinate names for SHOC datasets.
    #: The face coordinates are ``y_centre`` and ``x_centre``,
    #: left edge coordinates are ``y_left`` and ``x_left``,
    #: back edge coordinates are ``y_back`` and ``x_back``,
    #: and node coordinates are ``y_grid`` and ``x_grid``.
    coordinate_names = {
        ArakawaCGridKind.face: ('y_centre', 'x_centre'),
        ArakawaCGridKind.left: ('y_left', 'x_left'),
        ArakawaCGridKind.back: ('y_back', 'x_back'),
        ArakawaCGridKind.node: ('y_grid', 'x_grid'),
    }

    def get_depth_name(self) -> Hashable:
        name = 'z_centre'
        if name not in self.dataset.variables:
            raise NoSuchCoordinateError(
                f"SHOC dataset did not have expected depth coordinate {name!r}")
        return name

    def get_all_depth_names(self) -> List[Hashable]:
        return ['z_centre', 'z_grid']

    def get_time_name(self) -> Hashable:
        name = 't'
        if name not in self.dataset.variables:
            raise NoSuchCoordinateError(
                f"SHOC dataset did not have expected time coordinate {name!r}")
        return name

    def drop_geometry(self) -> xarray.Dataset:
        dataset = super().drop_geometry()
        dataset.attrs.pop('Conventions', None)
        return dataset


class ShocSimple(CFGrid2D):
    """
    SHOC standard datasets can be simplified down to SHOC simple datasets,
    which are :class:`~.grid.CFGrid2D` curvilinear grids.
    The latitude and longitude coordinate variables are named ``j`` and ``i``.
    Edge and node dimensions are dropped.
    """
    _dimensions: Tuple[Hashable, Hashable] = ('j', 'i')

    @cached_property
    def topology(self) -> CFGrid2DTopology:
        y_dimension, x_dimension = self._dimensions
        try:
            latitude = next(
                name for name, variable in self.dataset.variables.items()
                if variable.dims == self._dimensions
                and variable.attrs["standard_name"] == "latitude"
            )
            longitude = next(
                name for name, variable in self.dataset.variables.items()
                if variable.dims == self._dimensions
                and variable.attrs["standard_name"] == "longitude"
            )
        except StopIteration:
            raise ValueError("Could not find the necessary coordinate variables")

        return CFGrid2DTopology(self.dataset, latitude=latitude, longitude=longitude)

    @classmethod
    def check_dataset(cls, dataset: xarray.Dataset) -> Optional[int]:
        if 'ems_version' not in dataset.attrs:
            return None
        if not set(dataset.dims).issuperset(cls._dimensions):
            return None
        return Specificity.HIGH

    def get_time_name(self) -> Hashable:
        name = 'time'
        if name not in self.dataset.variables:
            raise NoSuchCoordinateError(
                f"SHOC dataset did not have expected time coordinate {name!r}")
        return name

    def get_depth_name(self) -> Hashable:
        name = 'zc'
        if name not in self.dataset.variables:
            raise NoSuchCoordinateError(
                f"SHOC dataset did not have expected depth coordinate {name!r}")
        return name

    def get_all_depth_names(self) -> List[Hashable]:
        return [self.get_depth_name()]
