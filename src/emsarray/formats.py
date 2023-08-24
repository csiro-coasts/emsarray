import warnings
from functools import wraps
from typing import Any, Optional, Type

import xarray

from emsarray import conventions

from .conventions import Convention, GridKind, Index, get_dataset_convention


def _warn_old_new(old: str, new: str, **kwargs: Any) -> None:
    warnings.warn(
        f"{old} has been renamed to {new}",
        category=DeprecationWarning,
        **kwargs)


@wraps(get_dataset_convention)
def get_file_format(dataset: xarray.Dataset, **kwargs: Any) -> Optional[Type[Convention]]:
    _warn_old_new(
        old="emsarray.formats.get_file_format",
        new="emsarray.conventions.get_dataset_convention",
        stacklevel=3)
    return get_dataset_convention(dataset, **kwargs)


class Format(Convention[GridKind, Index]):
    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        _warn_old_new(
            old="emsarray.formats.Format",
            new="emsarray.conventions.Convention",
            stacklevel=4)
        super().__init_subclass__(*args, **kwargs)


def __getattr__(name: str) -> Any:
    # This takes the place of `from emsarray.conventions import *`,
    # and will warn on any `from emsarray.formats import x, y, z` uses.
    if not name.startswith('__'):
        _warn_old_new(
            old=f'emsarray.formats.{name}',
            new=f'emsarray.conventions.{name}',
            stacklevel=3)
    return getattr(conventions, name)


# Even this module is deprecated! Warn about that too
_warn_old_new(old='emsarray.formats', new='emsarray.conventions', stacklevel=3)
