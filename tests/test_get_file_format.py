import pytest
import xarray as xr

import emsarray
from emsarray.formats import get_file_format
from emsarray.formats.grid import CFGrid1D, CFGrid2D
from emsarray.formats.shoc import ShocSimple, ShocStandard
from emsarray.formats.ugrid import UGrid


@pytest.mark.parametrize(['dataset', 'expected_class'], [
    ('cfgrid1d.nc', CFGrid1D),
    ('cfgrid2d.nc', CFGrid2D),
    ('shoc_standard.nc', ShocStandard),
])
def test_files(datasets, dataset, expected_class):
    dataset = xr.open_dataset(datasets / dataset)
    assert get_file_format(dataset) is expected_class


@pytest.mark.tutorial
@pytest.mark.parametrize(['dataset', 'expected_class'], [
    ('bran2020', CFGrid1D),
    ('gbr4', ShocSimple),
    ('austen', UGrid),
])
def test_tutorial_datasets(dataset, expected_class):
    dataset = emsarray.tutorial.open_dataset(dataset)
    assert get_file_format(dataset) is expected_class
