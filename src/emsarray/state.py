"""
Dataclass for containing state required for emsarray
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, Optional, cast

import xarray as xr

if TYPE_CHECKING:
    from emsarray.formats._base import Format


@dataclasses.dataclass
class State:
    """
    Keeps state for emsarray.
    Currently only used to allow binding Format instances to datasets
    to avoid format autodetection.
    """
    dataset: xr.Dataset
    format: Optional[Format] = None

    accessor_name: Final[str] = "_emsarray_state"

    @classmethod
    def get(cls, dataset: xr.Dataset) -> "State":
        """
        Get the state for a dataset,
        making an empty state if none exists.
        """
        return cast(State, getattr(dataset, State.accessor_name))

    def bind_format(self, format: Format) -> None:
        """
        Bind a Format instance to this Dataset.
        If the Dataset is already bound, an error is raised.
        """
        self.format = format

    def is_bound(self) -> bool:
        """
        Check if the Dataset has a bound format.
        """
        return self.format is not None
