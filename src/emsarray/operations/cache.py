"""
Operations for making caching keys for a given dataset.
"""
import hashlib
import marshal

import numpy
import xarray

import emsarray


def hash_attributes(hash: "hashlib._Hash", attributes: dict) -> None:
    """
    Updates the provided hash with with a marshal serialised byte representation of the given attribute dictionary.

    Parameters
    ----------
    hash : hashlib-style hash instance
        The hash instance to update with the given attribute dict.
        This must follow the interface defined in :mod:`hashlib`.
    attributes: dict
        Expects a marshal compatible dictionary.
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
    Updates the provided hash with with a utf-8 encoded byte representation of the provided string.

    Parameters
    ----------
    hash : hashlib-style hash instance
        The hash instance to update with the given attribute dict.
        This must follow the interface defined in :mod:`hashlib`.
    attributes: str
        Expects a string that can be encoded in utf-8.
    """
    # Prepend the str length
    hash_int(hash, len(value))
    hash.update(value.encode('utf-8'))


def hash_int(hash: "hashlib._Hash", value: int) -> None:
    """
    Updates the provided hash with an encoded byte representation of the provided int.

    Parameters
    ----------
    hash : hashlib-style hash instance
        The hash instance to update with the given attribute dict.
        This must follow the interface defined in :mod:`hashlib`.
    attributes: int
        Expects an int that can be represented in a numpy int32.
    """
    hash.update(numpy.int32(value).tobytes())


def make_cache_key(dataset: xarray.Dataset, hash: "hashlib._Hash | None" = None) -> str:
    """
    Generate a key suitable for caching data derived from the geometry of a dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to generate a cache key from.
    hash : hashlib._Hash
        An instance of a hashlib hash class.
        Defaults to `hashlib.blake2b`, which is secure enough and fast enough for most purposes.
        The hash algorithm does not need to be cryptographically secure,
        so faster algorithms such as `xxhash` can be swapped in if desired.

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
        hash = hashlib.blake2b(digest_size=32)

    dataset.ems.hash_geometry(hash)

    # Hash convention name, convention module path and emsarray version
    hash_string(hash, dataset.ems.__class__.__module__)
    hash_string(hash, dataset.ems.__class__.__name__)
    hash_string(hash, emsarray.__version__)

    return hash.hexdigest()
