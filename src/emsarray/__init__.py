"""
emsarray - xarray utilities for various non-CF grid data conventions
"""
import importlib.metadata
import warnings
from typing import Any

from . import tutorial
from .accessors import ems_accessor
from .conventions import Convention, get_dataset_convention, open_dataset

__version__ = importlib.metadata.version("emsarray")

__all__ = [
    "tutorial",
    "ems_accessor",
    "Convention", "get_dataset_convention", "open_dataset",
]


def __getattr__(name: str) -> Any:
    old_new = {
        'Format': ('Convention', 'emsarray.formats.Format'),
        'get_file_format': ('get_dataset_convention', 'emsarray.formats.get_file_format'),
    }
    if name in old_new:
        new, old_path = old_new[name]
        warnings.warn(
            f"emsarray.{name} has been renamed to emsarray.{new}",
            category=DeprecationWarning,
            stacklevel=2)
        old_module, old_name = old_path.rsplit('.', 1)
        return getattr(importlib.import_module(old_module), old_name)

    raise AttributeError(f"{name} is not defined")
