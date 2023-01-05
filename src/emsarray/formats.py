import warnings
from functools import wraps
from typing import Any, Optional, Type

import xarray as xr

from .conventions import *  # noqa
from .conventions import Convention, get_dataset_convention


def _warn_old_new(old: str, new: str, **kwargs: Any) -> None:
    warnings.warn(
        f"{old} has been renamed to {new}",
        category=DeprecationWarning,
        **kwargs)


@wraps(get_dataset_convention)
def get_file_format(dataset: xr.Dataset, **kwargs: Any) -> Optional[Type[Convention]]:
    _warn_old_new(
        old="emsarray.formats.get_file_format",
        new="emsarray.conventions.get_dataset_convention",
        stacklevel=3)
    return get_dataset_convention(dataset, **kwargs)


class Format(Convention):
    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        _warn_old_new(
            old="emsarray.formats.Format",
            new="emsarray.conventions.Convention",
            stacklevel=4)
        super().__init_subclass__(*args, **kwargs)


_warn_old_new(old="emsarray.formats", new="emsarray.conventions", stacklevel=3)
