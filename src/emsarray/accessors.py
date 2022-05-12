from __future__ import annotations

import xarray as xr

from .formats import Format, get_file_format


@xr.register_dataset_accessor("ems")
def ems_accessor(dataset: xr.Dataset) -> Format:
    """Provides the ``.ems`` attribute on xarray Datasets.
    This will make a :class:`~emsarray.formats.Format` instance for the dataset,
    using the correct :class:`~emsarray.formats.Format` subclass depending on the file type.

    Returns
    -------
    :class:`~emsarray.formats.Format`
    """
    format_class = get_file_format(dataset)
    if format_class is None:
        raise RuntimeError("Could not determine format of dataset")
    return format_class(dataset)
