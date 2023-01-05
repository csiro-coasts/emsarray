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
    Refer to that repository for more information,
    including where the datasets were sourced from
    and full licensing details.

    The example datasets that are available are:

    ``austen``
        A day of the AUSTEn National Tidal model data.
        This dataset is defined on an :class:`unstructured grid <.UGrid>`.

    ``bran2020``
        A small sample of the Bluelink Reanalysis 2020 (BRAN2020) ocean dataset.
        This dataset is defined on a rectangular grid with one dimensional coordinates,
        handled by the :class:`.CFGrid1D` convenion.

    ``fraser``
        A subset of the Great Barrier Reef 4km (GBR4) v2.0 model, part of the eReefs data.
        This subset is centred around K'gari / Fraser Island.
        This dataset is defined on a curvilinear grid with two dimensional coordinates,
        handled by the :class:`.CFGrid2D` convention.
        Temperature, sea surface height, and current variables are included.

    ``gbr4``
        A subset of the Great Barrier Reef 4km (GBR4) v2.0 model, part of the eReefs data.
        This dataset is defined on a curvilinear grid with two dimensional coordinates,
        handled by the :class:`.CFGrid2D` convention.
        Temperature, sea surface heigh, and salinity variables are included.

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

    Example
    -------

    .. code-block:: python

        import emsarray
        fraser = emsarray.tutorial.open_dataset('fraser')
        fraser.ems.plot(fraser['temp'].isel(time=0, k=-1))
    """
    local_path = _fetch(name)
    return emsarray.open_dataset(local_path, **kwargs)
