import abc
import dataclasses
import warnings
from typing import Callable

import numpy

from emsarray.conventions._base import Convention


@dataclasses.dataclass()
class Hotfix:
    hotfix_cls: type
    implements: set[str]
    warning: str

    def apply(self, convention_cls: type[Convention]) -> type[Convention]:
        warnings.warn(self.warning.format(convention=convention_cls.__name__))
        patched_cls = type(convention_cls.__name__, (self.hotfix_cls, convention_cls), {})
        abc.update_abstractmethods(patched_cls)
        return patched_cls


hotfixes: list[Hotfix] = []


def register_hotfix(
    implements: set[str],
    warning: str,
) -> Callable[[type], type]:
    def decorator(hotfix_cls: type) -> type:
        hotfixes.append(Hotfix(hotfix_cls=hotfix_cls, implements=implements, warning=warning))
        return hotfix_cls
    return decorator


def hotfix_convention(convention_cls: type[Convention]) -> type[Convention]:
    abstract_methods: frozenset[str] = getattr(convention_cls, '__abstractmethods__', frozenset())
    if not abstract_methods:
        return convention_cls

    to_apply = []
    for hotfix in hotfixes:
        if hotfix.implements.issubset(abstract_methods):
            to_apply.append(hotfix)
            abstract_methods = abstract_methods - hotfix.implements

    if abstract_methods:
        patched = convention_cls.__abstractmethods__ - abstract_methods
        raise Exception(
            f"Convention {convention_cls.__module__}.{convention_cls.__qualname__} "
            f"is missing implementations for methods {', '.join(convention_cls.__abstractmethods__)}. "
            f"Hotfixes were unavailable for methods {', '.join(sorted(patched))}."
        )

    for hotfix in to_apply:
        convention_cls = hotfix.apply(convention_cls)

    return convention_cls


@register_hotfix(
    {'_make_polygons'},
    "{convention} class implements `polygons`, which was renamed to `_make_polygons` in 0.8.0",
)
class MakePolygonHotfix:
    polygons = Convention.polygons

    def _make_polygons(self) -> numpy.ndarray:
        return super().polygons  # type: ignore
