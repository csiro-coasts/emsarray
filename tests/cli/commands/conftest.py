import pytest

from emsarray.cli import utils


@pytest.fixture(autouse=True)
def disable_logging_config(monkeypatch: pytest.MonkeyPatch):
    """
    Calling `set_verbosity` will change the logging config in ways that can
    spoil other unrelated tests that rely on the default log config.
    """
    monkeypatch.setattr(utils, 'set_verbosity', lambda x: None)
