import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_equal

from emsarray.operations import normalize_depth_variables


@pytest.mark.parametrize(
    "input_depths,input_positive,input_deep_to_shallow", [
        ([-1, +0, +1, +2, +3, +4], 'down', False),
        ([+4, +3, +2, +1, +0, -1], 'down', True),
        ([+1, +0, -1, -2, -3, -4], 'up', False),
        ([-4, -3, -2, -1, +0, +1], 'up', True),
    ],
)
@pytest.mark.parametrize("set_positive", [True, False])
@pytest.mark.parametrize(
    ["expected", "positive_down", "deep_to_shallow"], [
        ([+4, +3, +2, +1, +0, -1], True, True),
        ([-1, +0, +1, +2, +3, +4], True, False),
        ([-4, -3, -2, -1, +0, +1], False, True),
        ([+1, +0, -1, -2, -3, -4], False, False),
    ],
)
def test_normalize_depth_variable(
    input_depths: np.ndarray, input_positive: str, input_deep_to_shallow: bool,
    set_positive: bool,
    expected: np.ndarray, positive_down: bool, deep_to_shallow: bool,
    recwarn,
):
    input_depths = input_depths[:]
    # Some datasets have a coordinate with the same dimension name
    depth_coord = xr.DataArray(
        data=np.array(input_depths),
        dims=['depth_coord'],
        attrs={'positive': input_positive if set_positive else None, 'foo': 'bar'},
    )
    # Some dimensions have different coordinate and dimension names
    depth_name = xr.DataArray(
        data=np.array(input_depths),
        dims=['depth_dimension'],
        attrs={'positive': input_positive if set_positive else None, 'foo': 'bar'},
    )
    values = np.arange(4 * 6 * 4).reshape(4, 6, 4)
    dataset = xr.Dataset(
        data_vars={
            "values_coord": (["time", "depth_coord", "x"], values.copy()),
            "values_dimension": (["time", "depth_dimension", "x"], values.copy()),
        },
        coords={
            "time": (["time"], np.arange(4)),
            "depth_coord": depth_coord,
            "depth_name": depth_name,
            "location": (["location"], np.arange(4)),
        },
    )

    out = normalize_depth_variables(
        dataset, ['depth_coord', 'depth_name'],
        positive_down=positive_down, deep_to_shallow=deep_to_shallow,
    )

    # Check that the values are reordered along the depth axis if required
    expected_values = (
        values[:, :, :] if (input_deep_to_shallow == deep_to_shallow)
        else values[:, ::-1, :])
    assert_equal(out['values_coord'].values, expected_values)
    assert_equal(out['values_dimension'].values, expected_values)

    # Check that attributes on the depth coordinate were not clobbered
    assert out['depth_coord'].attrs == {
        'positive': 'down' if positive_down else 'up',
        'foo': 'bar',
    }
    assert out['depth_name'].attrs == {
        'positive': 'down' if positive_down else 'up',
        'foo': 'bar',
    }

    # Check the depth values are as expected
    assert_equal(out['depth_coord'].values, expected)
    assert out['depth_coord'].dims == ('depth_coord',)

    assert_equal(out['depth_name'].values, expected)
    assert out['depth_name'].dims == ('depth_dimension',)

    assert out.dims['depth_coord'] == 6
    assert out.dims['depth_dimension'] == 6

    # Check that a warning was raised if the positive: 'up'/'down' attribute
    # was not set
    if set_positive:
        assert len(recwarn) == 0
    else:
        assert len(recwarn) == 2
        assert f'`positive: {input_positive!r}`' in str(recwarn[0].message)
