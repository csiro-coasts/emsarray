from __future__ import annotations

import logging

import xarray

from .conventions import Convention, get_dataset_convention
from .state import State

logger = logging.getLogger(__name__)


@xarray.register_dataset_accessor("ems")
def ems_accessor(dataset: xarray.Dataset) -> Convention:
    """Provides the ``.ems`` attribute on xarray Datasets.
    This will make a :class:`~emsarray.conventions.Convention` instance for the dataset,
    using the correct :class:`~emsarray.conventions.Convention` subclass depending on the file type.

    Returns
    -------
    :class:`~emsarray.conventions.Convention`
    """
    state = State.get(dataset)
    if state.convention is not None:
        return state.convention

    convention_class = get_dataset_convention(dataset)
    if convention_class is None:
        raise RuntimeError("Could not determine convention of dataset")

    convention = convention_class(dataset)
    convention.bind()
    return convention


xarray.register_dataset_accessor(State.accessor_name)(State)
