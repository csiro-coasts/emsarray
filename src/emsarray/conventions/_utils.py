import logging
from typing import Any

import xarray

from emsarray.types import Pathish

logger = logging.getLogger(__name__)


def open_dataset(path: Pathish, **kwargs: Any) -> xarray.Dataset:
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

    See Also
    --------
    :func:`xarray.open_dataset`
    """
    dataset = xarray.open_dataset(path, **kwargs)

    # Determine the correct convention. All the magic happens in the accessor.
    convention = dataset.ems
    convention_class = type(convention)
    logger.debug(
        "Using convention %s.%s for dataset %r",
        convention_class.__module__, convention_class.__name__, str(path))

    return dataset
