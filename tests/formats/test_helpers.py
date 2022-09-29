"""
Test format class registrationby entry points or manual registration.
"""
import sys
import unittest.mock
from typing import List, Tuple

from emsarray.formats import (
    ArakawaC, CFGrid1D, CFGrid2D, ShocSimple, ShocStandard, UGrid, _helpers
)
from emsarray.formats._helpers import (
    FormatRegistry, entry_point_formats, register_format
)

EXPECTED_FORMATS = [
    ArakawaC, CFGrid1D, CFGrid2D, ShocSimple, ShocStandard, UGrid
]


def test_entry_point_formats():
    assert list(entry_point_formats()) == EXPECTED_FORMATS


def test_register_format_decorator(monkeypatch):
    test_registry = FormatRegistry()
    monkeypatch.setattr(_helpers, 'registry', test_registry)

    @register_format
    class Foo:
        pass

    # The decorator should return the class again
    assert Foo is not None
    assert isinstance(Foo, type)

    # The registered format should be in the registry
    assert Foo in test_registry.registered_formats


def test_format_registration():
    class Foo:
        pass

    test_registry = FormatRegistry()

    # No manually registered formats
    assert test_registry.registered_formats == []
    # Expected entry point formats
    assert test_registry.entry_point_formats == EXPECTED_FORMATS
    # Combination of the two
    assert test_registry.formats == EXPECTED_FORMATS

    # Register a format
    test_registry.add_format(Foo)
    assert test_registry.registered_formats == [Foo]

    # Manually registered formats should come first
    assert test_registry.formats == [Foo] + EXPECTED_FORMATS


def monkeypatch_entrypoint(
    monkeypatch,
    items: List[Tuple[str, str]],
    group='emsarray.formats',
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
        ('CFGrid1D', 'emsarray.formats.grid:CFGrid1D'),
    ])
    assert list(entry_point_formats()) == [CFGrid1D]
    assert len(caplog.records) == 0


def test_mock_entry_points_duplicate(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.formats.grid:CFGrid1D'),
        ('CFGrid1D', 'emsarray.formats.grid:CFGrid1D'),
        ('CFGrid2D', 'emsarray.formats.grid:CFGrid2D'),
    ])
    assert list(entry_point_formats()) == [CFGrid1D, CFGrid2D]
    assert len(caplog.records) == 0


def test_mock_entry_points_irrelevant_name(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('IrrelevantName', 'emsarray.formats.grid:CFGrid1D'),
        ('CFGrid2D', 'emsarray.formats.grid:CFGrid2D'),
    ])
    assert list(entry_point_formats()) == [CFGrid1D, CFGrid2D]
    assert len(caplog.records) == 0


def test_mock_entry_points_bad_module(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.formats.nope:CFGrid1D'),
        ('CFGrid2D', 'emsarray.formats.grid:CFGrid2D'),
    ])
    assert list(entry_point_formats()) == [CFGrid2D]

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert 'Error loading entry point' in record.message


def test_mock_entry_points_bad_attribute(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.formats.grid:Nope'),
        ('CFGrid2D', 'emsarray.formats.grid:CFGrid2D'),
    ])
    assert list(entry_point_formats()) == [CFGrid2D]

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert 'Error loading entry point' in record.message


def test_mock_entry_points_no_attr(caplog, monkeypatch):
    monkeypatch_entrypoint(monkeypatch, [
        ('CFGrid1D', 'emsarray.formats.grid'),
        ('CFGrid2D', 'emsarray.formats.grid:CFGrid2D'),
    ])
    assert list(entry_point_formats()) == [CFGrid2D]

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert record.message.startswith(
        'Entry point `CFGrid1D = emsarray.formats.grid` refers to <module')
