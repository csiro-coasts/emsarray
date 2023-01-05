from __future__ import annotations

import enum
import logging
import sys
from contextlib import suppress
from functools import cached_property
from itertools import chain
from typing import Any, Iterable, List, Optional, Tuple, Type

import xarray as xr

from emsarray.types import Pathish

from ._base import Convention

if sys.version_info >= (3, 10):
    from importlib import metadata
else:
    import importlib_metadata as metadata


logger = logging.getLogger(__name__)


class Specificity(enum.IntEnum):
    """
    How specific a match is when autodetecting a convention.
    Matches with higher specificity will be prioritised.

    General conventions such as CF Grid are low specificity,
    as many conventions extend and build on CF Grid conventions.

    The SHOC conventions extend the CF grid conventions,
    so a SHOC file will be detected as both CF Grid and SHOC.
    :class:`.ShocStandard` should return a higher specificity
    so that the correct convention implementation is used.
    """
    LOW = 10
    MEDIUM = 20
    HIGH = 30


class ConventionRegistry:
    registered_conventions: List[Type[Convention]]

    def __init__(self) -> None:
        self.registered_conventions = []

    @cached_property
    def conventions(self) -> Iterable[Type[Convention]]:
        """
        A list of all the registered Convention subclasses.
        This includes those registered via entry points
        and those registered via :func:`register_convention`.

        Returns
        -------
        list of Convention subclasses
            All the registered :class:`~emsarray.conventions.Convention` subclasses

        See also
        --------
        :func:`entry_point_conventions`
        :func:`register_convention`
        """
        conventions = []
        seen = set()
        # Construct a list of all registered conventions in the order that they
        # were registered. Manually registered conventions are prioritised.
        # Duplicates are removed.
        for convention in chain(self.registered_conventions, self.entry_point_conventions):
            if convention not in seen:
                conventions.append(convention)
                seen.add(convention)
        return conventions

    @cached_property
    def entry_point_conventions(self) -> List[Type[Convention]]:
        """
        Find all conventions registered via the ``emsarray.conventions`` entry point.
        This list is cached.

        Returns
        -------
        list of Convention subclasses
            All the :class:`~emsarray.conventions.Convention` subclasses registered
            via the ``emsarray.conventions`` entry point.
        """
        return list(entry_point_conventions())

    def add_convention(self, convention: Type[Convention]) -> None:
        """Register a Convention subclass with this registry.
        Datasets will be checked against this Convention when guessing file types.
        """
        with suppress(AttributeError):
            del self.conventions
        self.registered_conventions.append(convention)

    def match_conventions(self, dataset: xr.Dataset) -> List[Tuple[Type[Convention], int]]:
        """
        Get all :class:`~.Convention` implementations that support this dataset.

        Parameters
        ----------
        dataset : :class:`xarray.Dataset`
            The dataset to inspect

        Returns
        -------
        list of matches
            A list of ``(Convention class, match specificity)`` tuples.
            A higher specificity means a better match.
            The list of matches will be ordered from most to least specific.
        """
        matches: List[Tuple[Type[Convention], int]] = []
        for convention in self.conventions:
            match = convention.check_dataset(dataset)
            if match is not None:
                matches.append((convention, match))
        return sorted(matches, key=lambda m: m[1], reverse=True)

    def guess_convention(self, dataset: xr.Dataset) -> Optional[Type[Convention]]:
        """
        Guess the correct :class:`.Convention` implementation for a dataset.
        """
        matches = self.match_conventions(dataset)
        if matches:
            # Return the best match
            return matches[0][0]
        else:
            return None


registry = ConventionRegistry()


def get_dataset_convention(dataset: xr.Dataset) -> Optional[Type[Convention]]:
    """Find the most appropriate Convention subclass for this dataset.

    Parameters
    ----------
    dataset
        The dataset to introspect

    Returns
    -------
    :class:`.Convention`
        A :class:`.Convention` subclass appropriate for this dataset,
        or None if nothing appropriate can be found.
    """
    return registry.guess_convention(dataset)


def open_dataset(path: Pathish, **kwargs: Any) -> xr.Dataset:
    """
    Open a dataset and determine the correct Convention implementation for it.
    If a valid Convention implementation can not be found, an error is raised.

    Parameters
    ----------
    path
        The path to the dataset to open
    kwargs
        These are passed straight through to :func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset
        The opened dataset

    Example
    -------

    .. code-block:: python

        import emsarray
        dataset = emsarray.open_dataset("./tests/datasets/ugrid_mesh2d.nc")

    See also
    --------
    :meth:`emsarray.conventions.Convention.open_dataset`
    :func:`xarray.open_dataset`
    """
    dataset = xr.open_dataset(path, **kwargs)

    # Determine the correct convention. All the magic happens in the accessor.
    convention = dataset.ems
    convention_class = type(convention)
    logger.debug(
        "Using convention %s.%s for dataset %r",
        convention_class.__module__, convention_class.__name__, str(path))

    return dataset


def entry_point_conventions() -> Iterable[Type[Convention]]:
    """
    Finds conventions registered using entry points
    """
    seen = set()

    for entry_point in metadata.entry_points(group='emsarray.conventions'):
        try:
            obj = entry_point.load()
        except (AttributeError, ImportError):
            logger.exception("Error loading entry point %s", entry_point)
            continue

        if not (isinstance(obj, type) and issubclass(obj, Convention)):
            logger.error(
                "Entry point `%s = %s` refers to %r not a Convention subclass",
                entry_point.name, entry_point.value, obj)
            continue

        if obj not in seen:
            yield obj
            seen.add(obj)


def register_convention(convention: Type[Convention]) -> Type[Convention]:
    """
    Register a Convention subclass, used for guessing file types.
    Can be used as a decorator.

    This function is useful for making convention classes for internal project use.
    If you are distributing an emsarray convention class as a Python package,
    see :func:`entry_point_conventions` for registering conventions using an ``entry_point``.

    Example
    -------
    >>> from emsarray import conventions
    >>> @conventions.register
    ... class FooConvention(Convention):
    ...     pass

    See also
    --------
    :func:`entry_point_conventions`
    """
    registry.add_convention(convention)
    return convention
