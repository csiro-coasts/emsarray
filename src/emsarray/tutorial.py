"""
Useful functions for writing examples in the documentation
or writing tutorials for new users.
"""

import os
import pathlib
from typing import Any, cast

import xarray

import emsarray
from emsarray.utils import requires_extra

try:
    import pooch
    IMPORT_EXCEPTION = None
except ImportError as exc:
    IMPORT_EXCEPTION = exc


_requires_tutorial = requires_extra('tutorial', IMPORT_EXCEPTION)


BASE_URL = "https://github.com/csiro-coasts/emsarray-data/raw/{version}/{path}"

# Bump this version as new datasets are added or as old datasets are updated.
# Should be a tag name or commit hash, something immutable.
VERSION = "v0.3.0"


def _make_download_url(name: str) -> str:
    """URL to download a named file."""
    return BASE_URL.format(version=VERSION, path=name)


@_requires_tutorial
def _cache_directory() -> pathlib.Path:
    """Where to store cached downloads."""
    if 'EMSARRAY_DATA_DIR' in os.environ:
        path = pathlib.Path(os.environ['EMSARRAY_DATA_DIR'])
    else:
        path = cast(pathlib.Path, pooch.os_cache('emsarray_tutorial') / VERSION)
    path.mkdir(exist_ok=True, parents=True)
    return path


@_requires_tutorial
def _fetch(name: str, cache: bool = True) -> str:
    """
    Fetches a named file from the data repository to the local cache
    and returns the path to it.
    """
    logger = pooch.get_logger()
    logger.setLevel("WARNING")
    if not name.endswith('.nc'):
        name = name + '.nc'
    local_path = cast(str, pooch.retrieve(
        _make_download_url(name),
        known_hash=None,
        path=_cache_directory(),
    ))
    return local_path


@_requires_tutorial
def open_dataset(name: str, **kwargs: Any) -> xarray.Dataset:
    """
    Open one of the example datasets.

    Example datasets will be downloaded from
    the `dataset repository <https://github.com/csiro-coasts/emsarray-data/>`_
    and cached locally.

    The example datasets that are available are:

    TODO

    Parameters
    ----------
    name : str
        The name of the example dataset to open
    kwargs
        Passed through to :func:`emsarray.open_dataset`.

    Returns
    -------
    :class:`xarray.Dataset`
        The example dataset, after being downloaded and opened by xarray.
    """
    local_path = _fetch(name)
    return emsarray.open_dataset(local_path, **kwargs)
