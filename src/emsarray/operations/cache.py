"""
Operations for making cache keys based on dataset geometry.

Some operations such as :func:`~.operations.triangulate.triangulate_dataset`
only depend on the dataset geometry and are expensive to compute.
For applications that need to derive data from the dataset geometry
it would be useful if the derived data could be reused between different runs of the same application
or between multiple time slices of the same geometry distributed across multiple files.
This module provides :func:`.make_cache_key` to assist in this process
by deriving a cache key from the important parts of a dataset geometry.
Applications can use this cache key
as part of a filename when save derived geometry data to disk
or as a key to an in-memory cache of derived geometry.

The derived cache keys will be identical between different instances of an application,
and between different files in multi-file datasets split over an unlimited dimension.

This module does not provide an actual cache implementation.
"""
import hashlib
import marshal
from typing import cast

import numpy
import xarray

import emsarray


def hash_attributes(hash: "hashlib._Hash", attributes: dict) -> None:
    """
    Adds the contents of an :attr:`attributes dictionary <xarray.DataArray.attrs>`
    to a hash.

    Parameters
    ----------
    hash : hashlib-style hash instance
        The hash instance to add the attribute dictionary to.
        This must follow the interface defined in :mod:`hashlib`.
    attributes : dict
        A dictionary of attributes from a :class:`~xarray.Dataset` or :class:`~xarray.DataArray`.

    Notes
    -----
    The attribute dictionary is serialized to bytes using :func:`marshal.dumps`.
    This is an implementation detail that may change in future releases.
    """
    # Prepend the marshal encoding version
    marshal_version = 4
    hash_int(hash, marshal_version)
    # Specify marshal encoding version when serialising
    attribute_dict_marshal_bytes = marshal.dumps(attributes, marshal_version)
    # Prepend the number of attributes
    hash_int(hash, len(attributes))
    # Prepend the size of the pickled attributes
    hash_int(hash, len(attribute_dict_marshal_bytes))
    hash.update(attribute_dict_marshal_bytes)


def hash_string(hash: "hashlib._Hash", value: str) -> None:
    """
    Adds a :class:`string <str>` to a hash.

    Parameters
    ----------
    hash : hashlib-style hash instance
        The hash instance to add the string to.
        This must follow the interface defined in :mod:`hashlib`.
    value : str
        Any unicode string.

    Notes
    -----
    The string is UTF-8 encoded as part of being added to the hash.
    This is an implementation detail that may change in future releases.
    """
    # Prepend the length of the string to the hash
    # to prevent malicious datasets generating overlapping string hashes.
    hash_int(hash, len(value))
    hash.update(value.encode('utf-8'))


def hash_int(hash: "hashlib._Hash", value: int) -> None:
    """
    Adds an :class:`int` to a hash.

    Parameters
    ----------
    hash : hashlib-style hash instance
        The hash instance to add the integer to.
        This must follow the interface defined in :mod:`hashlib`.
    value : int
        Any int representable as an :data:`numpy.int32`

    Notes
    -----
    The int is cast to a :data:`numpy.int32` as part of being added to the hash.
    This is an implementation detail that may change in the future
    if larger integers are required.
    """
    with numpy.errstate(over='raise'):
        # Manual overflow check as older numpy versions dont throw the exception
        if numpy.iinfo("int32").min <= value <= numpy.iinfo("int32").max:
            hash.update(numpy.int32(value).tobytes())
        else:
            raise OverflowError


def make_cache_key(dataset: xarray.Dataset, hash: "hashlib._Hash | None" = None) -> str:
    """
    Derive a cache key from the geometry of a dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to generate a cache key from.
    hash : :mod:`hashlib`-compatible hash instance, optional
        An instance of a hashlib hash class.
        Defaults to :func:`hashlib.blake2b` with a digest size of 32,
        which is secure enough and fast enough for most purposes.

    Returns
    -------
    cache_key : str
        A string suitable for use as a cache key.
        The string will be safe for use as part of a filename if data is to be cached to disk.

    Examples
    --------

    .. code-block:: python

        import emsarray
        from emsarray.operations.cache import make_cache_key

        # Make a cache key from the dataset
        dataset = emsarray.tuorial.open_dataset("austen")
        cache_key = make_cache_key(dataset)
        >>> cache_key
        '580853c44e732878937598e86d0b26cb81e18d986072c0790a122244e9d3f480'

    Notes
    -----
    The cache key will depend on the Convention class,
    the emsarray version, and a hash of the geometry of the dataset.
    The specific structure of the cache key may change between emsarray and python versions,
    and should not be relied upon.
    """
    if hash is None:
        hash = cast("hashlib._Hash", hashlib.blake2b(digest_size=32))

    dataset.ems.hash_geometry(hash)

    # Hash convention name, convention module path and emsarray version
    hash_string(hash, dataset.ems.__class__.__module__)
    hash_string(hash, dataset.ems.__class__.__name__)
    hash_string(hash, emsarray.__version__)

    return hash.hexdigest()
