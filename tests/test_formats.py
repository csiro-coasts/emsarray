import importlib
import sys

import pytest
import xarray


@pytest.fixture(autouse=True)
def remove_cached_module():
    sys.modules.pop('emsarray.formats', None)


def test_warning_on_import():
    with pytest.warns(DeprecationWarning):
        importlib.import_module('emsarray.formats')


def test_warning_on_get_file_format(datasets):
    with pytest.warns(DeprecationWarning):
        from emsarray.formats import get_file_format

    dataset = xarray.open_dataset(datasets / 'cfgrid1d.nc')
    with pytest.warns(DeprecationWarning):
        get_file_format(dataset)


def test_warning_on_format_subclass(datasets):
    with pytest.warns(DeprecationWarning):
        from emsarray.formats import Format

    with pytest.warns(DeprecationWarning):
        class NewFormat(Format):
            pass
