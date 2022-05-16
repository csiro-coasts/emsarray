import pathlib

import pytest

import emsarray


@pytest.fixture
def datasets() -> pathlib.Path:
    here = pathlib.Path(__file__).parent
    return here / 'datasets'


def pytest_runtest_setup(item):
    if 'matplotlib' in item.keywords and "matplotlib_backend" not in item.fixturenames:
        item.fixturenames.append("matplotlib_backend")


@pytest.fixture
def matplotlib_backend(backend='template'):
    """
    Override the matplotlib backend for this test to 'template'
    """
    import matplotlib
    with matplotlib.rc_context({'backend': 'template'}):
        yield


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
