from typing import Optional

import numpy
import pytest
import xarray
from numpy.testing import assert_equal

from emsarray.operations.depth import normalize_depth_variables


@pytest.mark.parametrize(
    ["input_depths", "input_positive", "input_deep_to_shallow"], [
        ([-1, +0, +1, +2, +3, +4], 'down', False),
        ([+4, +3, +2, +1, +0, -1], 'down', True),
        ([+1, +0, -1, -2, -3, -4], 'up', False),
        ([-4, -3, -2, -1, +0, +1], 'up', True),
    ],
)
@pytest.mark.parametrize("set_positive", [True, False])
@pytest.mark.parametrize(
    ["positive_down", "deep_to_shallow"], [
        (None, None),
        (None, True),
        (None, False),
        (True, None),
        (True, True),
        (True, False),
        (False, None),
        (False, True),
        (False, False),
    ],
)
def test_normalize_depth_variable(
    input_depths: list[int],
    input_positive: str,
    input_deep_to_shallow: bool,
    set_positive: bool,
    positive_down: Optional[bool],
    deep_to_shallow: Optional[bool],
    recwarn,
):
    # Some datasets have a coordinate with the same dimension name
    positive_attr = {'positive': input_positive} if set_positive else {}
    depth_coord = xarray.DataArray(
        data=numpy.array(input_depths),
        dims=['depth_coord'],
        attrs={**positive_attr, 'foo': 'bar'},
    )
    # Some dimensions have different coordinate and dimension names
    depth_name = xarray.DataArray(
        data=numpy.array(input_depths),
        dims=['depth_dimension'],
        attrs={**positive_attr, 'foo': 'bar'},
    )
    values = numpy.arange(4 * 6 * 4).reshape(4, 6, 4)
    dataset = xarray.Dataset(
        data_vars={
            "values_coord": (["time", "depth_coord", "x"], values.copy()),
            "values_dimension": (["time", "depth_dimension", "x"], values.copy()),
        },
        coords={
            "time": (["time"], numpy.arange(4)),
            "depth_coord": depth_coord,
            "depth_name": depth_name,
            "location": (["location"], numpy.arange(4)),
        },
    )

    out = normalize_depth_variables(
        dataset,
        ['depth_coord', 'depth_name'],
        positive_down=positive_down,
        deep_to_shallow=deep_to_shallow,
    )

    expected_attrs = {'foo': 'bar'}
    if positive_down is None:
        if set_positive:
            expected_attrs['positive'] = input_positive
    elif positive_down:
        expected_attrs['positive'] = 'down'
    else:
        expected_attrs['positive'] = 'up'

    # Check that the values are reordered along the depth axis if required
    expected_values = values.copy()
    if deep_to_shallow is not None and input_deep_to_shallow != deep_to_shallow:
        expected_values = expected_values[:, ::-1, :]
    assert_equal(out['values_coord'].values, expected_values)
    assert_equal(out['values_dimension'].values, expected_values)

    # Check that attributes on the depth coordinate were not clobbered
    assert out['depth_coord'].attrs == expected_attrs
    assert out['depth_name'].attrs == expected_attrs

    # Check the depth values are as expected
    # The depths should be similar to the inputs,
    # possibly reversed and possibly negated depending on the input parameters.
    expected_depths = numpy.array(input_depths)
    if positive_down is not None and input_positive != expected_attrs['positive']:
        expected_depths = -expected_depths
    if deep_to_shallow is not None and input_deep_to_shallow != deep_to_shallow:
        expected_depths = expected_depths[::-1]

    assert_equal(out['depth_coord'].values, expected_depths)
    assert out['depth_coord'].dims == ('depth_coord',)

    assert_equal(out['depth_name'].values, expected_depths)
    assert out['depth_name'].dims == ('depth_dimension',)

    assert out.sizes['depth_coord'] == 6
    assert out.sizes['depth_dimension'] == 6

    # Check that a warning was raised if the positive: 'up'/'down' attribute
    # was not set
    if set_positive:
        assert len(recwarn) == 0
    else:
        assert len(recwarn) == 2
        assert f'`positive: {input_positive!r}`' in str(recwarn[0].message)
