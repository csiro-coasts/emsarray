from functools import cached_property
from typing import Any


def assert_property_not_cached(
    instance: Any,
    prop_name: str,
    /,
) -> None:
    __tracebackhide__ = True  # noqa
    cls = type(instance)
    prop = getattr(cls, prop_name)
    assert isinstance(prop, cached_property), \
        "{instance!r}.{prop_name} is not a cached_property"

    cache = instance.__dict__
    assert prop.attrname not in cache, \
        f"{instance!r}.{prop_name} was cached!"
