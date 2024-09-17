"""
Operations for making caching keys for given datasets and attribute dicts.
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
        Expects a marshal compatible dictionary
    """
    # Prepend the marshal encoding version
    marshal_version = 4
    hash.update(numpy.int32(marshal_version).tobytes('C'))
    # Specify marshal encoding version when serialising
    attribute_dict_marshal_bytes = marshal.dumps(attributes, marshal_version)
    # Prepend the number of attributes
    hash.update(numpy.int32(len(attributes)).tobytes('C'))
    # Prepend the size of the pickled attributes
    hash.update(numpy.int32(len(attribute_dict_marshal_bytes)).tobytes())
    hash.update(attribute_dict_marshal_bytes)


def make_cache_key(dataset: xarray.Dataset, hash: "hashlib._Hash | None" = None) -> str:
    """
    Generate a key suitable for caching data derived from the geometry of a dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to generate a cache key from
    hash : hash instance
        An instance of a hashlib hash class
        Defaults to `hashlib.blake2b`, which is secure enough and fast enough for most purposes.
        The hash algorithm does not need to be cryptographically secure,
        so faster algorithms such as `xxhash` can be swapped in if desired.

    Returns
    -------
    cache_key : str
        A string suitable for use as a cache key.
        The string will be safe for use as part filename if data are to be cached to disk.

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
    hash.update(dataset.ems.__class__.__module__.encode('utf-8'))
    hash.update(dataset.ems.__class__.__name__.encode('utf-8'))
    hash.update(emsarray.__version__.encode('utf-8'))

    return hash.hexdigest()
