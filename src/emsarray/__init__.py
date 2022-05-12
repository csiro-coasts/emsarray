"""
emsarray - xarray utilities for various non-CF grid data formats
"""

import importlib.metadata

from . import tutorial
from .accessors import ems_accessor
from .formats import Format, get_file_format, open_dataset

__version__ = importlib.metadata.version("emsarray")

__all__ = ["Format", "ems_accessor", "get_file_format", "open_dataset", "tutorial"]
