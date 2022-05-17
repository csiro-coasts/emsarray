import pytest

import emsarray
from emsarray.formats import get_file_format
from emsarray.formats.grid import CFGrid1D
from emsarray.formats.shoc import ShocSimple
from emsarray.formats.ugrid import UGrid


@pytest.mark.tutorial
@pytest.mark.parametrize(['dataset', 'expected_class'], [
    ('bran2020', CFGrid1D),
    ('gbr4', ShocSimple),
    ('austen', UGrid),
])
def test_tutorial_datasets(dataset, expected_class):
    dataset = emsarray.tutorial.open_dataset(dataset)
    assert get_file_format(dataset) is expected_class
