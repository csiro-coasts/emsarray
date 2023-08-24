import pytest
import xarray

import emsarray
from emsarray.conventions import get_dataset_convention
from emsarray.conventions.grid import CFGrid1D, CFGrid2D
from emsarray.conventions.shoc import ShocSimple, ShocStandard
from emsarray.conventions.ugrid import UGrid


@pytest.mark.parametrize(['dataset', 'expected_class'], [
    ('cfgrid1d.nc', CFGrid1D),
    ('cfgrid2d.nc', CFGrid2D),
    ('shoc_standard.nc', ShocStandard),
    ('ugrid_mesh2d.nc', UGrid),
])
def test_files(datasets, dataset, expected_class):
    dataset = xarray.open_dataset(datasets / dataset)
    assert get_dataset_convention(dataset) is expected_class


@pytest.mark.tutorial
@pytest.mark.parametrize(['dataset', 'expected_class'], [
    ('bran2020', CFGrid1D),
    ('gbr4', ShocSimple),
    ('austen', UGrid),
])
def test_tutorial_datasets(dataset, expected_class):
    dataset = emsarray.tutorial.open_dataset(dataset)
    assert get_dataset_convention(dataset) is expected_class
