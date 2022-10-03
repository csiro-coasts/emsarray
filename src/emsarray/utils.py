"""Utility functions for working with datasets.
These are low-level functions that apply fixes to datasets,
or provide functionality missing in xarray.
Most users will not have to call these functions directly.

See also
--------
:mod:`emsarray.operations`
"""
from __future__ import annotations

import datetime
import functools
import itertools
import logging
import textwrap
import time
from types import TracebackType
from typing import (
    Any, Callable, Hashable, Iterable, List, Literal, Mapping, MutableMapping,
    Optional, Tuple, Type, TypeVar, Union, cast
)

import cftime
import netCDF4
import numpy as np
import pandas as pd
import pytz
import xarray as xr
from packaging.version import Version
from xarray.coding import times
from xarray.core.common import contains_cftime_datetimes

from emsarray.types import Pathish

logger = logging.getLogger(__name__)

DEFAULT_CALENDAR = 'proleptic_gregorian'


_T = TypeVar("_T")
_Exception = TypeVar("_Exception", bound=BaseException)


class PerfTimer:
    __slots__ = ('_start', '_stop', 'running')

    _start: float
    _stop: float
    running: bool

    def __init__(self) -> None:
        self.running = False

    def __enter__(self) -> PerfTimer:
        if self.running:
            raise RuntimeError("Timer is already running")
        self.running = True
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[_Exception]],
        exc_value: Optional[_Exception],
        traceback: TracebackType
    ) -> Optional[bool]:
        self._stop = time.perf_counter()
        self.running = False
        return None

    @property
    def elapsed(self) -> float:
        if self.running:
            raise RuntimeError("Timer is currently running")
        if not hasattr(self, "_start"):
            raise RuntimeError("Timer has not started yet")
        return self._stop - self._start


def to_netcdf_with_fixes(
    dataset: xr.Dataset,
    path: Pathish,
    time_variable: Optional[Hashable] = None,
    **kwargs: Any,
) -> None:
    """Saves a :class:`xarray.Dataset` to a netCDF4 file,
    applies various fixes to make it compatible with CSIRO software.

    Specifically, this:

    * prevents superfluous ``_FillValue`` attributes being added
      using :func:`.utils.disable_default_fill_value`,
    * Reformats time units that :mod:`xarray` struggles with
      using :func:`.utils.fix_bad_time_units`,
    * Reformats time units after saving to make it compatible with EMS
      using :func:`.utils.fix_time_units_for_ems`

    Parameters
    ----------
    dataset
        The :class:`xarray.Dataset` to save
    path
        Where to save the dataset
    time_variable
        The name of the time variable which needs fixing.
        Optional, if not provided the time variable will not be fixed for EMS.
    kwargs
        Any extra kwargs are passed to :meth:`xarray.Dataset.to_netcdf()`
    """
    # Make a shallow copy of the dataset so we can modify it before saving.
    # A shallow copy only copies the structure and attributes, but leaves
    # the data alone.
    dataset = dataset.copy(deep=False)

    # Fix default xarray behaviour around automatic _FillValues
    disable_default_fill_value(dataset)

    # Some time units are formatted in a way that will cause xarray to fail
    # when saving the dataset. This fixes that formatting on the way out.
    fix_bad_time_units(dataset)

    dataset.to_netcdf(path, **kwargs)
    if time_variable is not None:
        fix_time_units_for_ems(path, time_variable)


def format_time_units_for_ems(units: str, calendar: Optional[str] = DEFAULT_CALENDAR) -> str:
    """
    Reformat a given time unit string to an EMS-compatible string. ``xarray``
    will always format time unit strings using ISO8601 strings with
    a ``T`` separator and no space before the timezone. EMS is unable to parse
    this, and needs spaces between the date, time, and timezone components.

    Parameters
    ----------
    units
        A CF 'units' description of a time variable.
    calendar
        A CF 'calendar' attribute. Defaults to `"proleptic_gregorian"`.

    Returns
    -------
    str
        A new CF 'units' string, representing the same time, but formatted for EMS.

    Example
    -------
    >>> format_time_units_for_ems("days since 1990-01-01T00:00:00+10:00")
    "days since 1990-01-01 00:00:00 +10:00"
    """
    period, date_string = cftime._datesplit(units)
    time_bits = cftime._parse_date(date_string.strip())
    offset_total = time_bits[-1]
    tzinfo = pytz.FixedOffset(offset_total)

    reference_datetime = cftime.num2pydate(0, units, calendar)
    # datetimes come out of num2pydate naive and in UTC
    # This will put them in the correct timezone
    offset_datetime = reference_datetime.replace(tzinfo=pytz.UTC).astimezone(tzinfo)

    offset_hours, offset_minutes = divmod(int(time_bits[-1]), 60)
    offset_string = f'{offset_hours:+d}:{offset_minutes:02d}'

    new_units = f'{period} since {offset_datetime:%Y-%m-%d %H:%M:%S} {offset_string}'

    # Do a quick check that the reference time comes out the same when parsed
    # by cftime. As we are not adjusting the time data itself, the reference
    # time must not change.
    if cftime.num2pydate(0, new_units, calendar) != reference_datetime:
        raise ValueError(
            "New units does not resolve to the same reference time! "
            f"Existing: {units!r}, new: {new_units!r}"
        )

    return new_units


def fix_time_units_for_ems(
    dataset_path: Pathish,
    variable_name: Hashable,
) -> None:
    """
    Updates time units in a file so they are compatible with EMS.
    EMS only supports parsing a subset of valid time unit strings.

    When saving :class:`xarray.Dataset` objects, any time-based variables will be
    saved with a ``units`` like ``"days since 1990-01-01T00:00:00+10:00"`` - a full
    ISO 8601 date string. EMS is old and grumpy, and only accepts time units
    with a format like ``"days since 1990-01-01 00:00:00 +10"``.

    This function will do an in-place update of the time variable units in
    a dataset on disk. It will not recalculate any values, merely update the
    attribute.

    Parameters
    ----------
    dataset_path
        The path to the dataset on disk.
    variable_name
        The name of the time variable in the dataset to fix.
    """

    with netCDF4.Dataset(dataset_path, 'r+') as dataset:
        variable = dataset.variables[variable_name]

        units = cast(str, variable.getncattr('units'))
        calendar = cast(str, variable.getncattr('calendar') or DEFAULT_CALENDAR)

        variable.setncattr('units', format_time_units_for_ems(units, calendar))

        dataset.sync()


def _get_variables(dataset_or_array: Union[xr.Dataset, xr.DataArray]) -> List[xr.Variable]:
    if isinstance(dataset_or_array, xr.Dataset):
        return list(dataset_or_array.variables.values())
    else:
        return [dataset_or_array.variable]


def disable_default_fill_value(dataset_or_array: Union[xr.Dataset, xr.DataArray]) -> None:
    """
    Update all variables on this dataset or data array and disable the
    automatic ``_FillValue`` :mod:`xarray` sets. An automatic fill value can spoil
    ``missing_value``, violate CF conventions for coordinates, and generally
    change a dataset that was loaded from disk in unintentional ways.

    Parameters
    ----------
    dataset_or_array
        The :class:`xarray.Dataset` or :class:`xarray.DataArray` to update
    """
    for variable in _get_variables(dataset_or_array):
        if (
            issubclass(variable.dtype.type, np.floating)
            and "_FillValue" not in variable.encoding
            and "_FillValue" not in variable.attrs
        ):
            variable.encoding["_FillValue"] = None


def fix_bad_time_units(dataset_or_array: Union[xr.Dataset, xr.DataArray]) -> None:
    """Some datasets have a time units string that causes xarray to raise an
    error when saving the dataset. The unit string is parsed when reading the
    dataset just fine, only saving is the issue. This function will check for
    these bad unit strings and change them in place to something xarray can
    handle.

    This issue was fixed in https://github.com/pydata/xarray/pull/6049
    and released in version 0.21.0.
    Once the minimum supported version of xarray is 0.21.0 or higher
    this entire function can be removed.
    """
    if Version(xr.__version__) >= Version('0.21.0'):
        return

    for variable in _get_variables(dataset_or_array):
        # This is the same check xarray uses in xarray.coding.times.CFDatetimeCoder
        is_datetime = (
            np.issubdtype(variable.data.dtype, np.datetime64)
            or contains_cftime_datetimes(variable)
        )
        if is_datetime and 'units' in variable.encoding:
            if 'units' not in variable.encoding:
                # xarray.open_mfdataset() does not propagate the `encoding` dict.
                # This is where the 'units' and 'calendar' attributes are stored.
                # We can't do anything here without those attributes.
                continue
            units = variable.encoding['units']
            calendar = variable.encoding.get('calendar')
            try:
                delta, timestamp = times._unpack_netcdf_time_units(units)
            except ValueError:
                continue  # Don't bother fixing this one - too broken

            try:
                pd.Timestamp(timestamp)
                continue  # These units are formatted fine and don't need fixing
            except ValueError:
                pass

            # This function uses cftime to parse the datestamp and reformat it
            # to something nice, which should take care of the issue for us.
            variable.encoding['units'] = format_time_units_for_ems(units, calendar)


def dataset_like(sample_dataset: xr.Dataset, new_dataset: xr.Dataset) -> xr.Dataset:
    """
    Take an example dataset, and another dataset with identical variable names
    and coordinates, and rearrange the new dataset to have identical ordering
    to the sample. Useful for making a multi-file dataset resemble a sample
    dataset, for example fter masking and saving each variable one-by-one to
    a file.

    Parameters
    ----------
    sample_dataset
        The :class:`xarray.Dataset` to copy the order and attributes from
    new_dataset
        The :class:`xarray.Dataset` to copy the data from.

    Returns
    -------
    :class:`xarray.Dataset`
        A new dataset with attributes and orderings taken from
        ``sample_dataset`` and data taken from ``new_dataset``.
    """
    like_dataset = xr.Dataset(
        # The keys are listed in-order in sample_dataset.
        # Remaking `new_dataset` using this order will fix most things
        data_vars={key: new_dataset[key] for key in sample_dataset.data_vars.keys()},
        coords={key: new_dataset[key] for key in sample_dataset.coords.keys()},
    )
    like_dataset.attrs = new_dataset.attrs.copy()
    like_dataset.encoding = new_dataset.encoding.copy()

    # Copy over attributes and encodings
    _update_no_clobber(sample_dataset.attrs, like_dataset.attrs)
    _update_no_clobber(sample_dataset.encoding, like_dataset.encoding)
    for key, sample_variable in sample_dataset.variables.items():
        new_variable = like_dataset.variables[key]
        _update_no_clobber(sample_variable.attrs, new_variable.attrs)
        _update_no_clobber(sample_variable.encoding, new_variable.encoding)

    # Done!
    return like_dataset


def _update_no_clobber(source: Mapping[Hashable, Any], dest: MutableMapping[Hashable, Any]) -> None:
    """
    Update `dest` dict with values from `source`, without clobbering existing
    keys in `dest`.
    """
    for key, value in source.items():
        if key not in dest:
            dest[key] = value


def extract_vars(
    dataset: xr.Dataset,
    variables: Iterable[Hashable],
    keep_bounds: bool = True,
    errors: Literal['raise', 'ignore'] = 'raise',
) -> xr.Dataset:
    """Extract a set of variables from a dataset, dropping all others.

    This is approximately the opposite of :meth:`xarray.Dataset.drop_vars`.

    Parameters
    ----------
    dataset
        The dataset to extract the variables from
    variables
        A list of variable names
    keep_bounds
        If true (the default), additionally keep any `bounds` variables for the
        included variables and all coordinates.
    errors : {"raise", "ignore"}, optional
        If 'raise' (default), raises a :exc:`ValueError` error if any of the
        variable passed are not in the dataset. If 'ignore', any given names
        that are in the dataset are kept and no error is raised.

    Returns
    -------
    :class:`xr.Dataset`
        A new dataset with only the named variables included.

    See also
    --------
    :meth:`xarray.Dataset.drop_vars`
    """
    variables = set(variables)

    if errors == 'raise':
        missing_variables = variables - set(dataset.variables.keys())
        if missing_variables:
            raise ValueError(
                f"Variables {sorted(missing_variables, key=str)!r} "
                "do not exist in the dataset"
            )

    if keep_bounds:
        variables_and_coordinates = variables | set(dataset.coords.keys())
        variables = variables | {
            dataset[var].attrs['bounds'] for var in variables_and_coordinates
            if 'bounds' in dataset[var].attrs
        }

    drop_vars = list(set(dataset.data_vars.keys()) - variables)
    drop_vars = [name for name in dataset.data_vars.keys() if name not in variables]
    logger.debug("Dropping variables %r", drop_vars)
    return dataset.drop_vars(drop_vars)


def pairwise(iterable: Iterable[_T]) -> Iterable[Tuple[_T, _T]]:
    """
    Iterate over values in an iterator in pairs.

    Example
    -------
    >>> for a, b in pairwise("ABCD"):
    ...     print(a, b)
    A B
    B C
    C D
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def dimensions_from_coords(
    dataset: xr.Dataset,
    coordinate_names: List[Hashable],
) -> List[Hashable]:
    """
    Get the names of the dimensions for a set of coordinates.

    Parameters
    ----------
    dataset
        The dataset to get the dimensions from
    coordinate_names
        The names of some coordinate variables.

    Returns
    -------
    list of Hashable
        The name of the relevant dimension for each coordinate variable.
    """
    dimensions = []
    for coordinate_name in coordinate_names:
        # Don't use dataset.coords[], auxilliary coordinates do not appear there
        data_array = dataset.variables[coordinate_name]
        if len(data_array.dims) > 1:
            raise ValueError(
                f"Coordinate variable {coordinate_name} has more "
                "than one dimension: {data_array.dims}")
        dimensions.append(data_array.dims[0])

    return dimensions


def check_data_array_dimensions_match(dataset: xr.Dataset, data_array: xr.DataArray) -> None:
    """
    Check that the dimensions of a :class:`xarray.DataArray`
    match the dimensions of a :class:`xarray.Dataset`.
    This is useful when using the metadata of a particular dataset to display a data array,
    without requiring the data array to be taken directly from the dataset.

    If the dimensions do not match, a ValueError is raised, indicating the mismatched dimension.

    Parameters
    ----------
    dataset
        The dataset used as a reference
    data_array
        The data array to check the dimensions of

    Raises
    ------
    ValueError
        Raised if the dimensions do not match
    """
    for dimension, data_array_size in zip(data_array.dims, data_array.shape):
        if dimension not in dataset.dims:
            raise ValueError(
                f"Data array has unknown dimension {dimension} of size {data_array_size}"
            )

        dataset_size = dataset.dims[dimension]
        if data_array_size != dataset_size:
            raise ValueError(
                "Dimension mismatch between dataset and data array: "
                f"Dataset dimension {dimension} has size {dataset_size}, "
                f"data array has size {data_array_size}"
            )


def move_dimensions_to_end(
    data_array: xr.DataArray,
    dimensions: List[Hashable],
) -> xr.DataArray:
    """
    Transpose the dimensions of a :class:`xarray.DataArray`
    such that the given dimensions appear as the last dimensions,
    in the order given.

    Other dimensions appear as the first dimensions,
    in the same order they are present in the original dataset

    Parameters
    ----------
    `data_array` : :class:`xarray.DataArray`
        The data array to transpose
    `dimensions` : list of Hashable
        The dimensions to move to the end

    Examples
    --------

    .. code-block:: python

        >>> data_array.dims
        ('a', 'b', 'c', 'd')
        >>> transposed = move_dimensions_to_end(data_array, ['c', 'b'])
        >>> transposed.dims
        ('a', 'd', 'c', 'b')
    """
    current_dims = set(data_array.dims)
    if not current_dims.issuperset(dimensions):
        missing = sorted(set(dimensions) - set(current_dims), key=str)
        raise ValueError(f"DataArray does not contain dimensions {missing!r}")

    new_order = [dim for dim in data_array.dims if dim not in dimensions] + dimensions
    if new_order == list(data_array.dims):
        # Don't bother transposing if the dimensions are already correct.
        return data_array.copy(deep=False)
    else:
        return data_array.transpose(*new_order)


def linearise_dimensions(
    data_array: xr.DataArray,
    dimensions: List[Hashable],
    linear_dimension: Optional[Hashable] = None,
) -> xr.DataArray:
    """
    Flatten the given dimensions of a :class:`~xarray.DataArray`.
    Other dimensions are kept as-is.
    This is useful for turning a DataArray with dimensions ('t', 'z', 'y', 'x')
    in to ('t', 'z', 'index').

    Parameters
    ----------

    `data_array` : :class:`xarray.DataArray`
        The data array to linearize
    `dimensions` : list of Hashable
        The dimensions to linearize, in the desired order.
        These dimensions can be in any order and any position in the input data array.
    `linear_dimension` : Hashable, optional
        The name of the new dimension of flattened data.
        Defaults to `index`, or `index_0`, `index_1`, etc if not given.

    Returns
    -------
    :class:`xarray.DataArray`
        A new data array with the specified dimensions flattened.
        Only data, coordinates, and dimensions are set,
        attributes and encodings are not copied over.

    Examples
    --------

    .. code-block:: python

        >>> data_array = xr.DataArray(
        ...     data=np.random.random((3, 5, 7)),
        ...     dims=['x', 'y', 'z'],
        ... )
        >>> flattened = linearise_dimensions(data_array, ['y', 'x'])
        >>> flattened.dims
        ('z', 'index')
        >>> flattened.shape
        (7, 15)
        >>> expected = np.transpose(data_array.isel(z=0).values).ravel()
        >>> all(flattened.isel(z=0).values == expected)
        True
    """
    data_array = move_dimensions_to_end(data_array, dimensions)
    new_shape = data_array.shape[:-len(dimensions)] + (-1,)
    new_data = data_array.values.reshape(new_shape)
    existing_dims = data_array.dims[:-len(dimensions)]

    if linear_dimension is None:
        suffix = 0
        linear_dimension = 'index'
        while linear_dimension in existing_dims:
            linear_dimension = f'index_{suffix}'
            suffix += 1
    new_dims = existing_dims + (linear_dimension,)

    coords = {
        name: coord for name, coord in data_array.coords.items()
        if set(coord.dims).issubset(existing_dims)}
    return xr.DataArray(data=new_data, dims=new_dims, coords=coords)


def datetime_from_np_time(np_time: np.datetime64) -> datetime.datetime:
    """
    Convert a numpy :class:`~numpy.datetime64`
    to a python :class:`~datetime.datetime`.
    Useful when formatting dates for human consumption,
    as a numpy :class:`~numpy.datetime64`
    has no equivalent of :meth:`datetime.datetime.strftime`.

    This does present the possibility of losing precision,
    as a numpy datetime64 has variable accuracy up to an attosecond,
    while Python datetimes have fixed microsecond accuracy.
    A conversion that truncates data is not reported as an error.
    If you're using numpy datetime64 with attosecond accuracy,
    the Python datetime formatting methods are insufficient for your needs anyway.
    """
    return datetime.datetime.fromtimestamp(np_time.item() / 10**9)


class RequiresExtraException(Exception):
    """
    Raised when the optional dependencies for some functionality have not been installed,
    and a function requiring them is called.

    See also
    --------
    :func:`requires_extra`
    """

    message_template = """
        emsarray must be installed with the '{extra}' extra in order to use this functionality:

            $ pip install emsarray[{extra}]

        Or, if installed using conda:

            $ conda install emsarray
    """
    message_template = textwrap.dedent(message_template).strip("\n")

    def __init__(self, extra: str) -> None:
        super().__init__(self.message_template.format(extra=extra))
        self.extra = extra


def requires_extra(
    extra: str,
    import_error: Optional[ImportError],
    exception_class: Type[RequiresExtraException] = RequiresExtraException,
) -> Callable[[_T], _T]:
    if import_error is None:
        return lambda fn: fn

    def error_decorator(fn: _T) -> _T:
        @functools.wraps(fn)  # type: ignore
        def error(*args: Any, **kwargs: Any) -> Any:
            raise exception_class(extra) from import_error
        return error  # type: ignore
    return error_decorator
