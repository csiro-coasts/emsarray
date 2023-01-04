from __future__ import annotations

import logging

import xarray as xr

from .formats import Format, get_file_format
from .state import State

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("ems")
def ems_accessor(dataset: xr.Dataset) -> Format:
    """Provides the ``.ems`` attribute on xarray Datasets.
    This will make a :class:`~emsarray.formats.Format` instance for the dataset,
    using the correct :class:`~emsarray.formats.Format` subclass depending on the file type.

    Returns
    -------
    :class:`~emsarray.formats.Format`
    """
    state = State.get(dataset)
    if state.format is not None:
        return state.format

    format_class = get_file_format(dataset)
    if format_class is None:
        raise RuntimeError("Could not determine format of dataset")

    format = format_class(dataset)
    format.bind()
    return format


xr.register_dataset_accessor(State.accessor_name)(State)
