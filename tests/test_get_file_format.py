import pytest
import xarray as xr

from emsarray.formats import get_file_format
from emsarray.formats.grid import CFGrid1D
from emsarray.formats.shoc import ShocSimple, ShocStandard
from emsarray.formats.ugrid import UGrid


@pytest.mark.skip(reason="Tutorial datasets are a work in progress")
def test_cfgrid():
    dataset = xr.tutorial.open_dataset('cfgrid_oceanmap')
    assert get_file_format(dataset) is CFGrid1D


@pytest.mark.skip(reason="Tutorial datasets are a work in progress")
def test_shoc_standard():
    dataset = xr.tutorial.open_dataset('shoc_standard')
    assert get_file_format(dataset) is ShocStandard


@pytest.mark.skip(reason="Tutorial datasets are a work in progress")
def test_shoc_simple():
    dataset = xr.tutorial.open_dataset('shoc_simple')
    assert get_file_format(dataset) is ShocSimple


@pytest.mark.skip(reason="Tutorial datasets are a work in progress")
def test_unstructured_grid():
    dataset = xr.tutorial.open_dataset('ugrid_mesh2d', mask_and_scale=False)
    assert get_file_format(dataset) is UGrid
