"""
Test binding Convention instances to datasets,
whether implicitly through the accessor autodetection or manually.
"""
import pytest
import xarray

import emsarray
from emsarray.conventions import CFGrid2D, ShocStandard
from emsarray.state import State


def test_automatic_binding_via_accessor(datasets):
    ds = xarray.open_dataset(datasets / 'cfgrid1d.nc')
    state = State.get(ds)

    # Fresh datasets opened via xarray should be unbound initially
    assert not state.is_bound()
    assert state.convention is None

    # Autodetection via the accessor should bind the convention
    convention = ds.ems
    assert state.is_bound()
    assert state.convention is convention


def test_automatic_binding_via_open_dataset(datasets):
    ds = emsarray.open_dataset(datasets / 'cfgrid1d.nc')
    state = State.get(ds)

    # Datasets opened via emsarray.open_dataset should be bound
    assert state.is_bound()
    assert ds.ems is state.convention


def test_manual_binding(datasets):
    ds = xarray.open_dataset(datasets / 'shoc_standard.nc')
    state = State.get(ds)

    # Fresh datasets opened via xarray should be unbound initially
    assert not state.is_bound()
    assert state.convention is None

    # Construct a convention. Autodetection would use ShocStandard,
    # but CFGrid2D is also compatible.
    convention = CFGrid2D(ds, longitude='x_centre', latitude='y_centre')

    # Merely constructing this convention should not bind anything
    assert not state.is_bound()
    assert state.convention is None

    # Manually bind and check that it holds
    convention.bind()
    assert state.is_bound()
    assert state.convention is convention
    assert ds.ems is convention


def test_copy_rebind(datasets):
    ds_shoc = emsarray.open_dataset(datasets / 'shoc_standard.nc')
    shoc_convention = ds_shoc.ems
    assert isinstance(shoc_convention, ShocStandard)

    # Binding an already bound dataset should raise an error
    cfgrid_convention = CFGrid2D(ds_shoc, longitude='x_centre', latitude='y_centre')
    with pytest.raises(ValueError):
        cfgrid_convention.bind()

    # Making a copy and binding that should work
    ds_cfgrid = ds_shoc.copy()
    cfgrid_convention = CFGrid2D(ds_cfgrid, longitude='x_centre', latitude='y_centre')
    cfgrid_convention.bind()

    # Each dataset should have its own bound convention
    assert ds_cfgrid.ems is cfgrid_convention
    assert ds_shoc.ems is shoc_convention
