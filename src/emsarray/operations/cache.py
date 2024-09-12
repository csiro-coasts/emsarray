"""
Operations for making a triangular mesh out of the polygons of a dataset.
"""
from __future__ import annotations

import hashlib
import inspect
import pickle
from typing import Any

import xarray


def hash_attributes(hash: hashlib._Hash, attribute_dict: xarray.Variable) -> None:
    """
    Update the provided hash with with a pickle serialised byte representation of the given attribute dictionary.

    Parameters
    ----------
    hash : hashlib-style hash instance
        The hash instance to update with geometry data.
        This must follow the interface defined in :mod:`hashlib`.
    attribute_dict: xarray.Variable dictionary
        Expects a pickleable xarray.Variable dictionary
    """
    # Specifying pickle protocol version to prevent hash changes when the default is updated
    attribute_dict_bytes = pickle.dumps(attribute_dict, protocol=5)
    hash.update(attribute_dict_bytes)


def make_cache_key(dataset: xarray.Dataset, hash_type: type[hashlib._Hash] | Any = hashlib.blake2b) -> str:
    """
    Generate a key suitable for caching data derived from the geometry of a dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to generate a cache key from
    hash : hash class
        The kind of hash to use.
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
    The specific structure of the cache key may change between emsarray versions
    and should not be relied upon.
    """
    hash_sig = inspect.signature(hash_type.__init__)

    if ('digest_size' in hash_sig.parameters):
        m = hash_type(digest_size=32)
    else:
        m = hash_type()

    dataset.ems.hash_geometry(m)

    geometry_names = dataset.ems.get_all_geometry_names()

    for geometry_name in geometry_names:
        hash_attributes(m, dataset.variables[geometry_name])

    return m.hexdigest()
