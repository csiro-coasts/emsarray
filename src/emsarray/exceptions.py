"""
Exception and warning classes for specific exceptions
relating to emsarray and datasets.
"""


class EmsarrayError(Exception):
    """
    Base class for all emsarray-specific exception classes.
    """


class ConventionViolationError(EmsarrayError):
    """
    A dataset violates its conventions in a way that is not recoverable.
    """


class ConventionViolationWarning(UserWarning):
    """
    A dataset violates its conventions in a way that we can handle.
    For example, an attribute has an invalid type,
    but is still interpretable.
    """


class NoSuchCoordinateError(KeyError, EmsarrayError):
    """
    Raised when a dataset does not have a particular coordinate,
    such as in :attr:`.Convention.time_coordinate` and
    :attr:`.Convention.depth_coordinate`.
    """
