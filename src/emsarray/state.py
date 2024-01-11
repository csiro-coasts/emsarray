"""
Dataclass for containing state required for emsarray
"""
import dataclasses
from typing import Final, Optional, cast

import xarray

from emsarray import conventions


@dataclasses.dataclass
class State:
    """
    Keeps state for emsarray.
    Currently only used to allow binding Convention instances to datasets
    to avoid convention autodetection.
    """
    dataset: xarray.Dataset
    convention: Optional['conventions.Convention'] = None

    accessor_name: Final[str] = "_emsarray_state"

    @classmethod
    def get(cls, dataset: xarray.Dataset) -> "State":
        """
        Get the state for a dataset,
        making an empty state if none exists.
        """
        return cast(State, getattr(dataset, State.accessor_name))

    def bind_convention(self, convention: 'conventions.Convention') -> None:
        """
        Bind a Convention instance to this Dataset.
        If the Dataset is already bound, an error is raised.
        """
        self.convention = convention

    def is_bound(self) -> bool:
        """
        Check if the Dataset has a bound convention.
        """
        return self.convention is not None
