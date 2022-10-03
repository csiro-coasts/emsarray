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
