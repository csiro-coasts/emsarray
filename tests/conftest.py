import pytest


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
