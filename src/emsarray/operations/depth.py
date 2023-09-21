"""
These operations function on datasets with a depth axis,
such as the output from ocean models.
"""
import warnings
from collections import defaultdict
from typing import Dict, FrozenSet, Hashable, List, Optional, cast

import numpy
import xarray

from emsarray import utils


def ocean_floor(
    dataset: xarray.Dataset,
    depth_variables: List[Hashable],
    *,
    non_spatial_variables: Optional[List[Hashable]] = None,
) -> xarray.Dataset:
    """Make a new :class:`xarray.Dataset` reduced along the given depth
    coordinates to only contain values along the ocean floor.

    Parameters
    ----------
    dataset
        The dataset to reduce.
    depth_variables
        The names of depth coordinate variables.
        For supported conventions, use :meth:`.Convention.get_all_depth_names()`.
    non_spatial_variables
        Optional.
        A list of the names of any non-spatial coordinate variables, such as time.
        The ocean floor is assumed to be static across non-spatial dimensions.
        For supported conventions, use :meth:`.Convention.get_time_name()`.

    Returns
    -------
    :class:`xarray.Dataset`
        A new dataset with values taken from the deepest data.

    Examples
    --------

    .. code-block:: python

        >>> dataset
        <xarray.Dataset>
        Dimensions:  (z: 5, y: 5, x: 5)
        Coordinates:
            lon      (x) int64 0 -1 -2 -3 -4
            lat      (y) int64 0 1 2 3 4
            depth    (z) float64 4.25 3.25 2.25 1.25 0.25
        Dimensions without coordinates: z, y, x
        Data variables:
            temp     (z, y, x) float64 0.0 nan nan nan nan nan ... 4.0 4.0 4.0 4.0 4.0
        >>> operations.ocean_floor(dataset, ['depth'])
        <xarray.Dataset>
        Dimensions:  (y: 5, x: 5)
        Coordinates:
            lon      (x) int64 0 -1 -2 -3 -4
            lat      (y) int64 0 1 2 3 4
        Dimensions without coordinates: y, x
        Data variables:
            temp     (y, x) float64 0.0 1.0 2.0 3.0 4.0 1.0 ... 4.0 4.0 4.0 4.0 4.0

    This operation is relatively efficient,
    but still involves masking every variable that includes a depth axis.
    Where possible, do any time and space slicing before calling this method,
    and drop any variables you are not interested in.

    .. code-block:: python

        >>> operations.ocean_floor(
        ...     big_dataset['temp'].isel(record=0).to_dataset(),
        ...     depth_variables=big_dataset.ems.get_all_depth_names())
        <xarray.Dataset>
        Dimensions:  (y: 5, x: 5)
        Coordinates:
            lon      (x) int64 0 -1 -2 -3 -4
            lat      (y) int64 0 1 2 3 4
        Dimensions without coordinates: y, x
        Data variables:
            temp     (y, x) float64 0.0 1.0 2.0 3.0 4.0 1.0 ... 4.0 4.0 4.0 4.0 4.0

    See Also
    --------
    :meth:`.Convention.ocean_floor`
    :meth:`.Convention.get_all_depth_names`
    :func:`.normalize_depth_variables`
    :func:`.utils.extract_vars`
    """
    # Consider both curvilinear SHOC datasets and UGRID COMPASS datasets.
    #
    # SHOC datasets have four 'grids': faces, left edges, back edges, and vertices.
    # Each of these grids has a two dimensional (j, i) index.
    #
    # COMPASS datasets have up to three 'grids': faces, edges, and nodes.
    # Each of these grids has a one dimensional index.
    #
    # Data variables might be defined on any one of these grids,
    # might have non-spatial (e.g. time) dimensions,
    # and might have a depth axis.
    # We assume that a data variable has at most one depth axis,
    # and that for a combination of depth dimension and spatial dimensions,
    # the ocean floor is static.

    dataset = normalize_depth_variables(
        dataset, depth_variables,
        positive_down=True, deep_to_shallow=False)

    if non_spatial_variables is None:
        non_spatial_variables = []

    # The name of all the relevant _dimensions_, not _coordinates_
    depth_dimensions = utils.dimensions_from_coords(dataset, depth_variables)
    non_spatial_dimensions = utils.dimensions_from_coords(dataset, non_spatial_variables)

    for depth_dimension in sorted(depth_dimensions, key=hash):
        dimension_sets: Dict[FrozenSet[Hashable], List[Hashable]] = defaultdict(list)
        for name, variable in dataset.data_vars.items():
            if depth_dimension not in variable.dims:
                continue  # Skip data variables without this depth dimension

            spatial_dimensions = frozenset(variable.dims).difference(
                {depth_dimension}, non_spatial_dimensions)
            if not spatial_dimensions:
                continue  # Skip data variables with no other spatial dimenions

            dimension_sets[spatial_dimensions].append(name)

        for spatial_dimensions, variable_names in dimension_sets.items():
            # We now have a set of spatial_dimenions,
            # and a list of data variable_names.
            # We assume that a specific combination of depth variable and
            # spatial dimensions has a static ocean floor.
            # We find the ocean floor for one of the data variables,
            # and then use that to mask out each data variable in turn.

            # Get an example data array and drop all the non-spatial dimensions.
            data_array = dataset.data_vars[variable_names[0]].isel(
                {name: 0 for name in non_spatial_dimensions},
                drop=True, missing_dims='ignore')
            # Then find the ocean floor indices.
            ocean_floor_indices = _find_ocean_floor_indices(
                data_array, depth_dimension)

            # Extract just the variables with these spatial coordinates
            dataset_subset = utils.extract_vars(dataset, variable_names)

            # Drop any coordinates for this depth variable.
            # For some reason .isel() call will play havok with them,
            # so best to drop them beforehand.
            dataset_subset = dataset_subset.drop_vars([
                name for name, coordinate in dataset_subset.coords.items()
                if coordinate.dims == (depth_dimension,)
            ])

            # Find the ocean floor using the ocean_floor_indices
            dataset_subset = dataset_subset.isel(
                {depth_dimension: ocean_floor_indices},
                drop=True, missing_dims='ignore')

            # Merge these floored variables back in to the dataset
            dataset = dataset_subset.merge(dataset, compat='override')

    # Finally, drop the depth dimensions.
    # This will clear up any residual variables that use the depth variables,
    # such as depth coordinate variables.
    # errors='ignore' because the depth dimensions may have already been dropped
    dataset = dataset.drop_dims(depth_dimensions, errors='ignore')

    return dataset


def _find_ocean_floor_indices(
    data_array: xarray.DataArray,
    depth_dimension: Hashable,
) -> xarray.DataArray:
    # This needs some explaining.
    # (any number * 0 + 1) is 1, while (nan * 0 + 1) is nan.
    # As layers under the ocean floor are nans,
    # this will give us a series of 1's for water layers, with nans below,
    # such as [1, 1, 1, nan, nan, nan].
    # `.cumsum()` will then add all the 1's up cumulatively,
    # giving something like `[1, 2, 3, nan, nan, nan]` for a column.
    # `.argmax()` will find the highest non-nan index, and we have our answer!
    #
    # Columns of all nans will have an argmax index of 0.
    # Item 0 in the column will be nan, resulting in nan in the output as desired.
    depth_indices = (data_array * 0 + 1).cumsum(str(depth_dimension))
    max_depth_indices = depth_indices.argmax(str(depth_dimension))
    return cast(xarray.DataArray, max_depth_indices)


def normalize_depth_variables(
    dataset: xarray.Dataset,
    depth_variables: List[Hashable],
    *,
    positive_down: Optional[bool] = None,
    deep_to_shallow: Optional[bool] = None,
) -> xarray.Dataset:
    """
    Some datasets represent depth as a positive variable, some as negative.
    Some datasets sort depth layers from deepest to most shallow, others
    from shallow to deep. :func:`normalize_depth_variables` will return
    a new dataset with the depth variables normalized.

    All depth variables should have a ``positive: "up"`` or ``positive: "down"`` attribute.
    If this attribute is missing,
    a warning is generated and
    a value is determined by examining the coordinate values.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to normalize
    depth_variables : list of Hashable
        The names of the depth coordinate variables.
        This should be the names of the variables, not the dimensions,
        for datasets where these differ.
    positive_down : bool, optional
        If True, positive values will indicate depth below the surface.
        If False, negative values indicate depth below the surface.
        If None, this attribute of the depth coordinate is left unmodified.
    deep_to_shallow : bool, optional
        If True, the layers are ordered such that deeper layers have lower indices.
        If False, the layers are ordered such that deeper layers have higher indices.
        If None, this attribute of the depth coordinate is left unmodified.

    Returns
    -------
    xarray.Dataset
        A copy of the dataset with the depth variables normalized.

    See Also
    --------
    :meth:`.Convention.normalize_depth_variables`
    :meth:`.Convention.get_all_depth_names`
    """
    new_dataset = dataset.copy()
    for name in depth_variables:
        variable = dataset[name]
        if len(variable.dims) != 1:
            raise ValueError(
                f"Can't normalize multidimensional depth variable {name!r} "
                f"with dimensions {list(variable.dims)!r}"
            )
        dimension = variable.dims[0]

        new_variable = new_dataset[name]
        if positive_down is not None:
            new_variable.attrs['positive'] = 'down' if positive_down else 'up'

        if 'positive' in variable.attrs:
            positive_attr = variable.attrs.get('positive')
            data_positive_down = (positive_attr == 'down')
        else:
            # No positive attribute set.
            # This is a violation of the CF conventions,
            # however it is a very common violation and we can make a good guess.
            # This is a _depth_ variable.
            # If there are more values >0 than <0, positive is probably down.
            total_values = len(variable.values)
            positive_values = len(variable.values[variable.values > 0])
            data_positive_down = positive_values > (total_values / 2)

            warnings.warn(
                f"Depth variable {name!r} had no 'positive' attribute, "
                f"guessing `positive: {'down' if data_positive_down else 'up'!r}`",
                stacklevel=2)

        if positive_down is not None and data_positive_down != positive_down:
            # Reverse the polarity
            new_values = -1 * new_variable.values
            if name == dimension:
                new_dataset = new_dataset.assign_coords({name: new_values})
                new_dataset[name].attrs = new_variable.attrs
                new_dataset[name].encoding = new_variable.encoding
                new_variable = new_dataset[name]
            else:
                new_dataset = new_dataset.assign({
                    name: ([dimension], new_values, new_variable.attrs, new_variable.encoding)
                })
                new_variable = new_dataset[name]

            try:
                bounds_name = new_variable.attrs['bounds']
                bounds_variable = new_dataset[bounds_name]
            except KeyError:
                pass
            else:
                new_dataset = new_dataset.assign({
                    bounds_name: (
                        bounds_variable.dims,
                        -1 * bounds_variable.values,
                        bounds_variable.attrs,
                        bounds_variable.encoding,
                    ),
                })

            # Update this so the deep-to-shallow normalization can use it
            data_positive_down = positive_down

        if deep_to_shallow is not None:
            # Check if the existing data goes from deep to shallow, correcting for
            # the positive_down we just adjusted above. This assumes that depth
            # data are monotonic across all values. If this is not the case,
            # good luck.
            d1, d2 = new_variable.values[0:2]
            data_deep_to_shallow = (d1 > d2) == data_positive_down

            # Flip the order of the coordinate
            if data_deep_to_shallow != deep_to_shallow:
                new_dataset = new_dataset.isel({dimension: numpy.s_[::-1]})

    return new_dataset
