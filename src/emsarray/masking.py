"""
Common functions for working with dataset masks.
Masks are used when clipping datasets to a smaller geographic subset,
such as :meth:`.Convention.clip`.
"""
from __future__ import annotations

import functools
import itertools
import logging
import operator
import pathlib
from typing import Any, Dict, Hashable, List, cast

import numpy
import xarray
from xarray.core.dtypes import maybe_promote

from emsarray import utils
from emsarray.types import Pathish

logger = logging.getLogger(__name__)


def mask_grid_dataset(
    dataset: xarray.Dataset,
    mask: xarray.Dataset,
    work_dir: Pathish,
    **kwargs: Any,
) -> xarray.Dataset:
    """Apply a mask to a two-dimensional grid dataset,
    such as :class:`.CFGrid1D` and :class:`.CFGrid2D`,
    or datasets with multiple grids such as :class:`.ArakawaC`

    Parameters
    ----------
    dataset
        The :class:`~xarray.Dataset` instance to mask
    mask
        The mask to apply. Different types of datasets need different masks.
    work_dir
        An empty directory where temporary files can be stored while applying
        the mask. The returned dataset will be built from files inside this
        directory, so callers must save the returned dataset before deleting
        this directory.
    kwargs
        Any extra kwargs are passed to `open_mfdataset` when assembling the
        new, clipped dataset.

    Returns
    -------
    :class:`~xarray.Dataset`
        The masked dataset
    """
    work_path = pathlib.Path(work_dir)

    logger.info("Calculating mask bounds...")
    # Trim the dataset and mask down to the new bounds. Doing this before
    # applying the mask is considerably faster
    bounds = calculate_grid_mask_bounds(mask)
    logger.debug("Bounds: %r", bounds)

    logger.info("Slicing mask and dataset to bounds...")
    mask = mask.isel(bounds)
    dataset = dataset.isel(bounds)

    mfdataset_names: List[pathlib.Path] = []

    logger.info("Applying masks...")
    # This is done variable-by-variable, as trying to do it to the entire
    # dataset is very memory intensive. This allows us to offload data to the
    # file system, at the added expense of having to recombine the dataset
    # afterwards.
    for key, data_array in dataset.data_vars.items():
        masked_data_array = mask_grid_data_array(mask, data_array)
        variable_path = work_path / f"{key}.nc"
        mfdataset_names.append(variable_path)
        utils.to_netcdf_with_fixes(masked_data_array.to_dataset(name=key), variable_path)
        del masked_data_array

    # Coordinates are handled differently. They have been trimmed already, but
    # shouldn't be masked. They are combined in to one dataset and saved as-is
    coords_path = work_path / "__coords__.nc"
    mfdataset_names.append(coords_path)
    utils.to_netcdf_with_fixes(xarray.Dataset(coords=dataset.coords), coords_path)

    logger.info("Merging individual variables")
    # Happily `mfdataset` will load data in to memory in a lazy manner,
    # allowing us to combine very large datasets without running out of memory.
    merged_dataset = xarray.open_mfdataset(
        mfdataset_names,
        # `lock=False` prevents an issue where opening a dataset with
        # `open_mfdataset` then saving it with `.to_netcdf()` would
        # intermittently hang.
        #
        # See https://github.com/pydata/xarray/issues/3961
        lock=False,
        **kwargs,
    )

    return utils.dataset_like(dataset, merged_dataset)


def mask_grid_data_array(mask: xarray.Dataset, data_array: xarray.DataArray) -> xarray.DataArray:
    """
    Apply a mask to a single data array.
    A mask dataset contains one or more mask data arrays.
    The mask to apply is selected by comparing dimensions
    - the first mask found which has dimensions
    that are a subset of the data array dimensions is used.

    Parameters
    ----------
    mask : xarray.Dataset
        The mask dataset
    data_array : xarray.DataArray
        The :class:`~xarray.DataArray` to mask

    Returns
    -------
    xarray.DataArray
        A new :class:`~xarray.DataArray`
        with any masked values replaced with _FillValue.
        The returned data array will be the same shape as the input data array.
        If no appropriate mask is found,
        the original data array is returned unmodified.
    """
    dimensions = set(data_array.dims)

    try:
        fill_value = find_fill_value(data_array)
    except ValueError:
        logger.debug(
            "Data array %r has no valid fill value, leaving as is",
            data_array.name)
        return data_array

    # Loop through each possible mask
    for mask_name, mask_data_array in mask.data_vars.items():
        # If every dimension of this mask exists in the data array, apply it
        if dimensions >= set(mask_data_array.dims):
            logger.debug(
                "Masking data array %r with mask %r",
                data_array.name, mask_name)
            new_data_array = cast(xarray.DataArray, data_array.where(mask_data_array, other=fill_value))
            new_data_array.attrs = data_array.attrs
            new_data_array.encoding = data_array.encoding
            return new_data_array

    # Fallback, no appropriate mask was found, so don't apply any.
    # This generally happens for data arrays such as time, record, x_grid, etc.
    logger.debug(
        "Data array %r had no relevant mask, leaving as is",
        data_array.name)
    return data_array


def find_fill_value(data_array: xarray.DataArray) -> Any:
    """
    Float-typed variables can easily be masked. If they don't already have
    a fill value, they can be masked using `NaN` without issue.
    However there are some `int`-typed variables without a fill value that
    _cant_ be automatically masked.

    Parameters
    ----------
    data_array : xarray.DataArray
        The :class:`~xarray.DataArray` to find an appropriate ``_FillValue`` for.

    Returns
    -------
    fill value
        A numpy scalar value appropriate to use
        as the fill value in the data array.

        * For masked arrays, this will be :data:`numpy.ma.masked` ---
          note that xarray itself does not use masked arrays, but is compatible with them.
        * For data arrays that already have a ``_FillValue`` used by xarray,
          :data:`numpy.nan` is returned.
          xarray will substitute in all ``_FillValue``
          with :data:`numpy.nan` when opening files.
        * For data arrays that have been opened with ``mask_and_scale=False``,
          the existing ``_FillValue`` is returned.
        * If the data array has a float ``dtype``,
          :data:`numpy.nan` is returned.
        * If none of the above are true, a :exc:`ValueError` is raised.

    """
    if numpy.ma.is_masked(data_array.values):
        # xarray does not use masked arrays, but just in case someone has
        # constructed a dataset using one...
        return numpy.ma.masked

    attrs = ['_FillValue', 'missing_value']
    for attr in attrs:
        if attr in data_array.attrs:
            # The dataset was opened with mask_and_scale=False and a mask has not
            # been applied. Masked values should be represented using _FillValue/missing_value.
            return data_array.attrs[attr]

    promoted_dtype, fill_value = maybe_promote(data_array.dtype)
    if promoted_dtype == data_array.dtype:
        return fill_value

    raise ValueError("No appropriate fill value found")


def calculate_grid_mask_bounds(mask: xarray.Dataset) -> Dict[Hashable, slice]:
    """
    Calculate the included bounds of a mask dataset for each dimension.

    Parameters
    ----------
    mask : xarray.Dataset
        The mask dataset should contain one or more boolean data arrays.

    Returns
    -------
    dict
        A dict of ``{dimension_name: slice(min_index, max_index)}`` will be returned.
        This dict can be passed directly in to a :meth:`xarray.Dataset.isel` call
        to crop a dataset to the bounds of a mask.
    """
    bounds = {}
    for name, mask_data_array in mask.data_vars.items():
        # Assert there is at least one true value somewhere in the entire mask
        # If this is False, strange errors would happen.
        if not mask_data_array.any().item():
            raise ValueError(f"Mask {name!r} is completely empty!")

        # Find the bounds for each dimension
        dimensions_set = set(mask_data_array.dims)
        for dimension in mask_data_array.dims:
            # For each step along this dimension, see if there are any True
            # cells in any of the other dimensions
            values = mask_data_array.any(dim=list(dimensions_set - {dimension}))

            # Find the first and last True values
            min_index = next(i for i, value in enumerate(values) if value)
            max_index = next(len(values) - i for i, value in enumerate(reversed(values)) if value)

            # max_index will actually be the index one _past_ the final True
            # value, but as slice ends are not inclusive, this is the desired
            # behaviour.
            bounds[dimension] = slice(min_index, max_index)

    return bounds


def smear_mask(arr: numpy.ndarray, pad_axes: List[bool]) -> numpy.ndarray:
    """
    Take a boolean numpy array and a list indicating which axes to smear along.
    Return a new array, expanded along the axes, with the boolean values
    smeared accordingly.

    This is a half baked convolution operator where the `pad_axes` parameter is
    used to build the kernel.

    Parameters
    ----------
    arr : numpy.ndarray
        A boolean numpy :class:`numpy.ndarray`.
    pad_axes : list of bool
        A list of booleans, indicating which axes to smear along.

    Returns
    -------
    numpy.ndarray
        The smeared array.
        For every axis where ``pad_axes`` was True,
        the array will be one element larger.

    Examples
    --------

    .. code-block:: pycon

        >>> arr
        array([[0, 0, 1, 0, 0],
               [0, 1, 0, 1, 0],
               [1, 0, 0, 0, 1]]

    Smear along the y-axis:

    .. code-block:: pycon

        >>> smear_mask(arr, [False, True])
        array([[0, 0, 1, 1, 0, 0],
               [0, 1, 1, 1, 1, 0],
               [1, 1, 0, 0, 1, 1]]

    Smear along both axes:

    .. code-block:: pycon

        >>> smear_mask(arr, [True, True])
        array([[0, 0, 1, 1, 0, 0],
               [0, 1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 0, 0, 1, 1])
    """
    paddings = itertools.product(*(
        [(1, 0), (0, 1)] if pad_axis else [(0, 0)]
        for pad_axis in pad_axes
    ))
    return functools.reduce(operator.or_, (numpy.pad(arr, pad) for pad in paddings))


def blur_mask(arr: numpy.ndarray, size: int = 1) -> numpy.ndarray:
    """
    Take a boolean numpy array and blur it, such that all indices neighbouring
    a True value in the input array are True in the output array. The output
    array will have the same shape as the input array.

    Parameters
    ----------
    arr : numpy.ndarray
        A boolean array to blur
    size : int
        The kernel size to use when blurring.
        In the output array,
        any cell that has a true value
        within its `size` neighbours along any axis
        is true.

    Returns
    -------
    numpy.ndarray
        The blurred array

    Examples
    --------

    .. code-block:: python

        >>> arr = numpy.array([
        ...     [1, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0],
        ...     [0, 0, 0, 1, 0],
        ...     [0, 0, 0, 0, 1]])
        >>> blur_mask(arr)
        array([[1, 1, 0, 0, 0],
               [1, 1, 1, 1, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 1, 1, 1]]
    """
    # Pad the mask with a `size` sized buffer.
    # This allows simple slicing to pull out a rectangular region around an
    # index, without worrying about the edges of the array.
    padded = numpy.pad(arr, size, constant_values=False)

    # For each cell in the original mask shape,
    # the blurred mask is true if the original mask was true,
    # or any cells in a `size` sized slice around the original cell.
    arr_iter = numpy.nditer(arr, ['multi_index'])
    indices = (arr_iter.multi_index for _ in arr_iter)
    values = (
        arr[index] or numpy.any(padded[tuple(slice(i, i + size * 2 + 1) for i in index)])
        for index in indices
    )

    arr = numpy.fromiter(values, count=arr.size, dtype=arr.dtype).reshape(arr.shape)
    return cast(numpy.ndarray, arr)
