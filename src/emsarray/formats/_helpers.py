from __future__ import annotations

import enum
import logging
from typing import Any, List, Optional, Tuple, Type

import xarray as xr

from emsarray.types import Pathish

from ._base import Format

logger = logging.getLogger(__name__)


class Specificity(enum.IntEnum):
    """
    How specific a match is when autodetecting a format.
    Matches with higher specificity will be prioritised.

    General formats such as CF Grid are low specificity,
    as many formats extend and build on CF Grid conventions.

    The SHOC conventions extend the CF grid conventions,
    so a SHOC file will be detected as both CF Grid and SHOC.
    :class:`ShocStandard` should return a higher specificity
    so that the correct format implementation is used.
    """
    LOW = 10
    MEDIUM = 20
    HIGH = 30


class FormatRegistry:
    formats: List[Type[Format]]

    def __init__(self) -> None:
        self.formats = []

    def add_format(self, format: Type[Format]) -> None:
        """Register a Format subclass with this registry.
        Datasets will be checked against this Format when guessing file types.
        """
        self.formats.append(format)

    def match_formats(self, dataset: xr.Dataset) -> List[Tuple[Type[Format], int]]:
        """
        Get all :class:`~.Format` implementations that support this dataset.

        Parameters
        ----------
        dataset : :class:`xarray.Dataset`
            The dataset to inspect

        Returns
        -------
        list of matches
            A list of ``(Format class, match specificity)`` tuples.
            A higher specificity means a better match.
            The list of matches will be ordered from most to least specific.
        """
        matches: List[Tuple[Type[Format], int]] = []
        for format in self.formats:
            match = format.check_dataset(dataset)
            if match is not None:
                matches.append((format, match))
        return sorted(matches, key=lambda m: m[1], reverse=True)

    def guess_format(self, dataset: xr.Dataset) -> Optional[Type[Format]]:
        """
        Guess the correct :class:`.Format` implementation for a dataset.
        """
        matches = self.match_formats(dataset)
        if matches:
            # Return the best match
            return matches[0][0]
        else:
            return None


registry = FormatRegistry()


def get_file_format(dataset: xr.Dataset) -> Optional[Type[Format]]:
    """Find the most appropriate Format subclass for this dataset.

    Parameters
    ----------
    dataset
        The dataset to introspect

    Returns
    -------
    :class:`.Format`
        A :class:`.Format` subclass appropriate for this dataset,
        or None if nothing appropriate can be found.
    """
    return registry.guess_format(dataset)


def open_dataset(path: Pathish, **kwargs: Any) -> xr.Dataset:
    """
    Determine the format of a dataset and then open it,
    setting any flags required by the format.

    Some dataset formats require certain flags to be set when opening a dataset.
    For example, :class:`~.ugrid.UGrid` datasets must be opened with ``scale_and_mask=False``.
    :func:`emsarray.open_dataset` delegates to the correct
    :class:`~emsarray.formats.Format` impelementation.

    Parameters
    ----------
    path
        The path to the dataset to open
    kwargs
        These are passed straight through to :func:`xarray.open_dataset`.

    Returns
    -------
    :class:`xarray.Dataset`
        The opened dataset

    Example
    -------

    .. code-block:: python

        import emsarray
        dataset = emsarray.open_dataset("./tests/datasets/ugrid_mesh2d.nc")

    See also
    --------
    :meth:`emsarray.formats.Format.open_dataset`
    :func:`xarray.open_dataset`

    """
    dataset = xr.open_dataset(path, **kwargs)
    file_format = get_file_format(dataset)
    if file_format is None:
        raise ValueError("Could not determine format of dataset {str(path)!r}")
    logger.debug("Opening dataset %r with %s", str(path), file_format.__name__)
    return file_format.open_dataset(path, **kwargs)


def register_format(format: Type[Format]) -> Type[Format]:
    """Register a Format subclass, used for guessing file types.
    Can be used as a decorator

    Example
    -------
    >>> from emsarray import formats
    >>> @formats.register
    ... class FooFormat(Format):
    ...     pass
    """
    registry.add_format(format)
    return format
