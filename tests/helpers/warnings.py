import contextlib
import warnings


@contextlib.contextmanager
def filter_warning(*args, record: bool = False, **kwargs):
    """
    A shortcut wrapper around warnings.catch_warning()
    and warnings.filterwarnings()
    """
    with warnings.catch_warnings(record=record) as context:
        warnings.filterwarnings(*args, **kwargs)
        yield context
