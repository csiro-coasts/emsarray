"""
Test binding Format instances to datasets,
whether implicitly through the accessor autodetection or manually.
"""
import pytest
import xarray as xr

import emsarray
from emsarray.formats import CFGrid2D, ShocStandard
from emsarray.state import State


def test_automatic_binding_via_accessor(datasets):
    ds = xr.open_dataset(datasets / 'cfgrid1d.nc')
    state = State.get(ds)

    # Fresh datasets opened via xarray should be unbound initially
    assert not state.is_bound()
    assert state.format is None

    # Autodetection via the accessor should bind the format
    format = ds.ems
    assert state.is_bound()
    assert state.format is format


def test_automatic_binding_via_open_dataset(datasets):
    ds = emsarray.open_dataset(datasets / 'cfgrid1d.nc')
    state = State.get(ds)

    # Datasets opened via emsarray.open_dataset should be bound
    assert state.is_bound()
    assert ds.ems is state.format


def test_manual_binding(datasets):
    ds = xr.open_dataset(datasets / 'shoc_standard.nc')
    state = State.get(ds)

    # Fresh datasets opened via xarray should be unbound initially
    assert not state.is_bound()
    assert state.format is None

    # Construct a format. Autodetection would use ShocStandard,
    # but CFGrid2D is also compatible.
    format = CFGrid2D(ds, longitude='x_centre', latitude='y_centre')

    # Merely constructing this format should not bind anything
    assert not state.is_bound()
    assert state.format is None

    # Manually bind and check that it holds
    format.bind()
    assert state.is_bound()
    assert state.format is format
    assert ds.ems is format


def test_copy_rebind(datasets):
    ds_shoc = emsarray.open_dataset(datasets / 'shoc_standard.nc')
    shoc_format = ds_shoc.ems
    assert isinstance(shoc_format, ShocStandard)

    # Binding an already bound dataset should raise an error
    cfgrid_format = CFGrid2D(ds_shoc, longitude='x_centre', latitude='y_centre')
    with pytest.raises(ValueError):
        cfgrid_format.bind()

    # Making a copy and binding that should work
    ds_cfgrid = ds_shoc.copy()
    cfgrid_format = CFGrid2D(ds_cfgrid, longitude='x_centre', latitude='y_centre')
    cfgrid_format.bind()

    # Each dataset should have its own bound format
    assert ds_cfgrid.ems is cfgrid_format
    assert ds_shoc.ems is shoc_format
