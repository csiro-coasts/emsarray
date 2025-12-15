"""
emsarray test suite.

.. option --dask-scheduler <scheduler>

   Set the scheduler for dask.
   Currently the test suite segfaults when the default scheduler is used.
   The scheduler is overridden to be 'synchronous', which avoids this issue.
   Use this flag to choose a different scheduler for testing fixes to the segfault.
"""
import logging
import pathlib
from unittest import mock

import dask
import pytest

import emsarray

logger = logging.getLogger(__name__)


@pytest.fixture
def datasets() -> pathlib.Path:
    here = pathlib.Path(__file__).parent
    return here / 'datasets'


def pytest_runtest_setup(item):
    if 'matplotlib' in item.keywords and "matplotlib_backend" not in item.fixturenames:
        item.fixturenames.append("matplotlib_backend")


def pytest_addoption(parser):
    parser.addoption(
        "--dask-scheduler", type=str, action="store", default="synchronous",
        help=(
            "Set the dask scheduler. Valid options include `synchronous' (the default), "
            "`distributed', `multiprocessing', `processes', `single-threaded', "
            "`sync', `synchronous', `threading', `threads'."
        ))


@pytest.fixture(autouse=True, scope='session')
def disable_dask_threads(request):
    """
    Currently the tests will regularly segfault while subsetting ugrid datasets.
    This only happens when using the latest dependencies installed from PyPI.
    Using older dependencies from PyPI or using latest dependencies from
    conda-forge continues to work fine.

    Disabling dask multithreading stops the issue.
    This is a temporary work around while the issue is investigated.

    To restore the default behaviour switch to the 'threads' scheduler:

        $ pytest --dask-scheduler 'threads'

    See also
    --------
    :option:`--dask-scheduler`
    https://github.com/csiro-coasts/emsarray/issues/139
    """
    dask.config.set(scheduler=request.config.option.dask_scheduler)


@pytest.fixture
def matplotlib_backend(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    backend: str = 'template',
):
    """
    Override the matplotlib backend for this test to 'template'
    """
    import matplotlib
    import matplotlib.pyplot

    node = request.node
    mark = node.get_closest_marker('matplotlib')

    if mark.kwargs.get('mock_coast', False):
        monkeypatch.setattr(emsarray.plot, 'add_coast', lambda figure: None)
        monkeypatch.setattr(emsarray.plot.shortcuts, 'add_coast', lambda figure: None)

    show_mock = mock.Mock(spec=matplotlib.pyplot.show)
    monkeypatch.setattr(matplotlib.pyplot, 'show', show_mock)

    with matplotlib.rc_context({'backend': 'template'}):
        yield
        matplotlib.pyplot.close('all')


def _needs_tutorial_mark(*args, **kwargs):
    """Helper function used in :func:`monkeypatch_tutorial`."""
    __tracebackhide__ = True  # Hide traceback in pytest
    pytest.fail(
        "Tests that use tutorial functions must be marked "
        "with @pytest.mark.tutorial"
    )


@pytest.fixture(autouse=True)
def monkeypatch_tutorial(request, monkeypatch):
    """
    By default, accessing the `emsarray.tutorial` functions are disabled in
    tests unless the tests have the `tutorial` mark. The tutorial datasets are
    fetched over the network and are a few megabytes each, so this saves some
    surprise downloads.
    """
    if 'tutorial' not in request.node.keywords:
        monkeypatch.setattr(emsarray.tutorial, 'open_dataset', _needs_tutorial_mark)
