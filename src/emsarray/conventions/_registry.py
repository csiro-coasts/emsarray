import logging
from collections.abc import Iterable
from contextlib import suppress
from functools import cached_property
from importlib import metadata
from itertools import chain

import xarray

from ._base import Convention

logger = logging.getLogger(__name__)


class ConventionRegistry:
    registered_conventions: list[type[Convention]]

    def __init__(self) -> None:
        self.registered_conventions = []

    @cached_property
    def conventions(self) -> Iterable[type[Convention]]:
        """
        A list of all the registered Convention subclasses.
        This includes those registered via entry points
        and those registered via :func:`register_convention`.

        Returns
        -------
        list of Convention subclasses
            All the registered :class:`~emsarray.conventions.Convention` subclasses

        See Also
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
    def entry_point_conventions(self) -> list[type[Convention]]:
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

    def add_convention(self, convention: type[Convention]) -> None:
        """Register a Convention subclass with this registry.
        Datasets will be checked against this Convention when guessing file types.
        """
        with suppress(AttributeError):
            del self.conventions
        self.registered_conventions.append(convention)

    def match_conventions(self, dataset: xarray.Dataset) -> list[tuple[type[Convention], int]]:
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
        matches: list[tuple[type[Convention], int]] = []
        for convention in self.conventions:
            match = convention.check_dataset(dataset)
            if match is not None:
                matches.append((convention, match))
        return sorted(matches, key=lambda m: m[1], reverse=True)

    def guess_convention(self, dataset: xarray.Dataset) -> type[Convention] | None:
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


def get_dataset_convention(dataset: xarray.Dataset) -> type[Convention] | None:
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

    Example
    -------

    .. code-block:: python

        >>> ds = emsarray.tutorial.open_dataset('austen.nc')
        >>> get_dataset_convention(ds)
        emsarray.conventions.ugrid.UGrid
    """
    return registry.guess_convention(dataset)


def entry_point_conventions() -> Iterable[type[Convention]]:
    """
    Finds conventions registered using entry points
    """
    seen = set()
    entry_points = metadata.entry_points(group='emsarray.conventions')
    for entry_point in entry_points:
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


def register_convention(convention: type[Convention]) -> type[Convention]:
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

    See Also
    --------
    :func:`entry_point_conventions`
    """
    registry.add_convention(convention)
    return convention
