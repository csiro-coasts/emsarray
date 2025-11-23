"""
emsarray - xarray utilities for various non-CF grid data conventions
"""
import importlib.metadata

from . import tutorial
from .accessors import ems_accessor
from .conventions import Convention, get_dataset_convention, open_dataset

__version__ = importlib.metadata.version("emsarray")

__all__ = [
    "tutorial",
    "ems_accessor",
    "Convention", "get_dataset_convention", "open_dataset",
]
