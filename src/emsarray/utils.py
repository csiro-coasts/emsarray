"""Utility functions for working with datasets.
These are low-level functions that apply fixes to datasets,
or provide functionality missing in xarray.
Most users will not have to call these functions directly.

See Also
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
import warnings
from types import TracebackType
from typing import (
    Any, Callable, Hashable, Iterable, List, Literal, Mapping, MutableMapping,
    Optional, Sequence, Tuple, Type, TypeVar, Union, cast
)

import cftime
import netCDF4
import numpy
import pytz
import shapely
import xarray
from xarray.core.dtypes import maybe_promote

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


def timed_func(fn: Callable[..., _T]) -> Callable[..., _T]:
    """
    Log the execution time of the decorated function.
    Logs "Calling ``<func.__qualname__>``" before the wrapped function is called,
    and "Completed ``<func.__qualname__>`` in ``<time>``s" after.
    The name of the logger is taken from ``func.__module__``.

    Example
    -------

    .. code-block:: python

        class Grass(Convention):
            @cached_property
            @timed_func
            def polygons(self):
                return ...

    When called, this will log something like::

        DEBUG Calling Grass.polygons
        DEBUG Completed Grass.polygons in 3.14s
    """
    fn_logger = logging.getLogger(fn.__module__)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        fn_logger.debug("Calling %s", fn.__qualname__)
        with PerfTimer() as timer:
            value = fn(*args, **kwargs)
        fn_logger.debug("Completed %s in %fs", fn.__qualname__, timer.elapsed)
        return value
    return wrapper


def to_netcdf_with_fixes(
    dataset: xarray.Dataset,
    path: Pathish,
    time_variable: Optional[Hashable] = None,
    **kwargs: Any,
) -> None:
    """Saves a :class:`xarray.Dataset` to a netCDF4 file,
    applies various fixes to make it compatible with CSIRO software.

    Specifically, this:

    * prevents superfluous ``_FillValue`` attributes being added
      using :func:`.utils.disable_default_fill_value`,
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


def _get_variables(dataset_or_array: Union[xarray.Dataset, xarray.DataArray]) -> List[xarray.Variable]:
    if isinstance(dataset_or_array, xarray.Dataset):
        return list(dataset_or_array.variables.values())
    else:
        return [dataset_or_array.variable]


def disable_default_fill_value(dataset_or_array: Union[xarray.Dataset, xarray.DataArray]) -> None:
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
        current_dtype = variable.dtype
        promoted_dtype, fill_value = maybe_promote(current_dtype)
        if (
            current_dtype == promoted_dtype
            and "_FillValue" not in variable.encoding
            and "_FillValue" not in variable.attrs
        ):
            variable.encoding["_FillValue"] = None


def dataset_like(sample_dataset: xarray.Dataset, new_dataset: xarray.Dataset) -> xarray.Dataset:
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
    like_dataset = xarray.Dataset(
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
    dataset: xarray.Dataset,
    variables: Iterable[Hashable],
    keep_bounds: bool = True,
    errors: Literal['raise', 'ignore'] = 'raise',
) -> xarray.Dataset:
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
    :class:`xarray.Dataset`
        A new dataset with only the named variables included.

    See Also
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
    dataset: xarray.Dataset,
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


def check_data_array_dimensions_match(dataset: xarray.Dataset, data_array: xarray.DataArray) -> None:
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
    data_array: xarray.DataArray,
    dimensions: List[Hashable],
) -> xarray.DataArray:
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


def ravel_dimensions(
    data_array: xarray.DataArray,
    dimensions: List[Hashable],
    linear_dimension: Optional[Hashable] = None,
) -> xarray.DataArray:
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

        >>> data_array = xarray.DataArray(
        ...     data=numpy.random.random((3, 5, 7)),
        ...     dims=['x', 'y', 'z'],
        ... )
        >>> flattened = ravel_dimensions(data_array, ['y', 'x'])
        >>> flattened.dims
        ('z', 'index')
        >>> flattened.shape
        (7, 15)
        >>> expected = numpy.transpose(data_array.isel(z=0).values).ravel()
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
    return xarray.DataArray(data=new_data, dims=new_dims, coords=coords)


def wind_dimension(
    data_array: xarray.DataArray,
    dimensions: Sequence[Hashable],
    sizes: Sequence[int],
    *,
    linear_dimension: Hashable = 'index',
) -> xarray.DataArray:
    """
    Replace a dimension in a data array by reshaping it in to one or more other dimensions.

    Parameters
    ----------
    data_array : xarray.DataArray
        The data array to reshape
    dimensions : sequence of Hashable
        The names of the new dimensions after reshaping.
    sizes : sequence of int
        The sizes of the new dimensions.
        The product of these sizes
        should match the size of the dimension being reshaped.
    linear_dimension : Hashable
        The name of the dimension to reshape.
        Defaults to 'index',
        the default name for linear dimensions returned by :func:`.ravel_dimensions`.

    Returns
    -------
    xarray.DataArray
        The original data array,
        with the linear dimension reshaped in to the new dimensions.

    Examples
    --------
    .. code-block:: python

        >>> data_array = xarray.DataArray(
        ...     data=numpy.arange(11 * 7 * 5 * 3).reshape(11, -1, 3),
        ...     dims=('time', 'index', 'colour'),
        ... )
        >>> data_array.sizes
        Frozen({'time': 11, 'index': 35, 'colour': 3})
        >>> wound_array = wind_dimensions(data_array, ['y', 'x'], [7, 5])
        >>> wound_array.sizes
        Frozen({'time': 11, 'y': 7, 'x': 5, 'colour': 3})

    See Also
    --------
    ravel_dimensions : The inverse operation
    """
    dimension_index = data_array.dims.index(linear_dimension)
    new_dims = splice_tuple(data_array.dims, dimension_index, dimensions)
    new_shape = splice_tuple(data_array.shape, dimension_index, sizes)
    new_data = data_array.values.reshape(new_shape)
    return xarray.DataArray(data=new_data, dims=new_dims)


def datetime_from_np_time(np_time: numpy.datetime64) -> datetime.datetime:
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

    See Also
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


def make_polygons_with_holes(
    points: numpy.ndarray,
    *,
    out: Optional[numpy.ndarray] = None,
) -> numpy.ndarray:
    """
    Make a :class:`numpy.ndarray` of :class:`shapely.Polygon` from an array of (n, m, 2) points.
    ``n`` is the number of polygons, ``m`` is the number of vertices per polygon.
    If any point in a polygon is :data:`numpy.nan`,
    that Polygon is skipped and will be :class:`None` in the returned array.

    Parameters
    ----------

    points : numpy.ndarray
        A (n, m, 2) array. Each row represents the m points of a polygon.
    out : numpy.ndarray, optional
        Optional. An array to fill with polygons.

    Returns
    -------

    numpy.ndarray
        The polygons in a array of size n.
    """
    if out is None:
        out = numpy.full(points.shape[0], None, dtype=numpy.object_)

    complete_row_indices = numpy.flatnonzero(numpy.isfinite(points).all(axis=(1, 2)))
    shapely.polygons(
        points[complete_row_indices],
        indices=complete_row_indices,
        out=out)
    return out


def deprecated(message: str, category: Type[Warning] = DeprecationWarning) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, category=category, stacklevel=2)
            return fn(*args, **kwargs)
        return wrapped
    return decorator


def splice_tuple(t: Tuple, index: int, values: Sequence) -> Tuple:
    return t[:index] + tuple(values) + t[index:][1:]
