import functools
from typing import Any, Callable, Protocol, TYPE_CHECKING

import xarray

if TYPE_CHECKING:
    from emsarray.conventions import Convention
t

def preserves_convention(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(dataset: xarray.Dataset, *args: Any, **kwargs: Any) -> xarray.Dataset:
        convention: Convention = dataset.ems
        return convention.copy_convention(func(dataset, *args, **kwargs))
    return wrapper


def preserves_topology(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(dataset: xarray.Dataset, *args: Any, **kwargs: Any) -> xarray.Dataset:
        convention: Convention = dataset.ems
        return convention.copy_topology(func(dataset, *args, **kwargs))
    return wrapper
