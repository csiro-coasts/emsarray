"""
Test convention class registration by entry points or manual registration.
"""
import sys
import unittest.mock
from typing import List, Tuple

from emsarray.conventions import (
    ArakawaC, CFGrid1D, CFGrid2D, ShocSimple, ShocStandard, UGrid, _helpers
)
from emsarray.conventions._helpers import (
    ConventionRegistry, entry_point_conventions, register_convention
)

EXPECTED_CONVENTIONS = [
    ArakawaC, CFGrid1D, CFGrid2D, ShocSimple, ShocStandard, UGrid
]


def test_entry_point_convetions():
    assert list(entry_point_conventions()) == EXPECTED_CONVENTIONS


def test_register_convention_decorator(monkeypatch):
    test_registry = ConventionRegistry()
    monkeypatch.setattr(_helpers, 'registry', test_registry)

    @register_convention
    class Foo:
        pass

    # The decorator should return the class again
    assert Foo is not None
    assert isinstance(Foo, type)

    # The registered convention should be in the registry
    assert Foo in test_registry.registered_conventions


def test_convention_registration():
    class Foo:
        pass

    test_registry = ConventionRegistry()

    # No manually registered conventions
    assert test_registry.registered_conventions == []
    # Expected entry point conventions
    assert test_registry.entry_point_conventions == EXPECTED_CONVENTIONS
    # Combination of the two
    assert test_registry.conventions == EXPECTED_CONVENTIONS

    # Register a convention
    test_registry.add_convention(Foo)
    assert test_registry.registered_conventions == [Foo]

    # Manually registered conventions should come first
    assert test_registry.conventions == [Foo] + EXPECTED_CONVENTIONS


def monkeypatch_entrypoint(
    monkeypatch,
    items: List[Tuple[str, str]],
    group='emsarray.conventions',
):
    if sys.version_info >= (3, 10):
        from importlib import metadata
    else:
        import importlib_metadata as metadata

    entry_points = [
        metadata.EntryPoint(group=group, name=name, value=value)
        for name, value in items
    ]
    mock = unittest.mock.create_autospec(
        metadata.entry_points, return_value=entry_points)
    monkeypatch.setattr(metadata, 'entry_points', mock)
    return entry_points


def test_mock_entry_points(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.conventions.grid:CFGrid1D'),
    ])
    assert list(entry_point_conventions()) == [CFGrid1D]
    assert len(caplog.records) == 0


def test_mock_entry_points_duplicate(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.conventions.grid:CFGrid1D'),
        ('CFGrid1D', 'emsarray.conventions.grid:CFGrid1D'),
        ('CFGrid2D', 'emsarray.conventions.grid:CFGrid2D'),
    ])
    assert list(entry_point_conventions()) == [CFGrid1D, CFGrid2D]
    assert len(caplog.records) == 0


def test_mock_entry_points_irrelevant_name(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('IrrelevantName', 'emsarray.conventions.grid:CFGrid1D'),
        ('CFGrid2D', 'emsarray.conventions.grid:CFGrid2D'),
    ])
    assert list(entry_point_conventions()) == [CFGrid1D, CFGrid2D]
    assert len(caplog.records) == 0


def test_mock_entry_points_bad_module(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.conventions.nope:CFGrid1D'),
        ('CFGrid2D', 'emsarray.conventions.grid:CFGrid2D'),
    ])
    assert list(entry_point_conventions()) == [CFGrid2D]

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert 'Error loading entry point' in record.message


def test_mock_entry_points_bad_attribute(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.conventions.grid:Nope'),
        ('CFGrid2D', 'emsarray.conventions.grid:CFGrid2D'),
    ])
    assert list(entry_point_conventions()) == [CFGrid2D]

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert 'Error loading entry point' in record.message


def test_mock_entry_points_no_attr(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.conventions.grid'),
        ('CFGrid2D', 'emsarray.conventions.grid:CFGrid2D'),
    ])
    assert list(entry_point_conventions()) == [CFGrid2D]

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert record.message.startswith(
        'Entry point `CFGrid1D = emsarray.conventions.grid` refers to <module')
