import argparse
import json
from pathlib import Path
from typing import Any

import geojson
import pytest
from shapely.geometry import box

from emsarray.cli import CommandException, utils


def test_nice_console_errors_no_errors(caplog: pytest.LogCaptureFixture) -> None:
    with utils.nice_console_errors():
        pass
    assert len(caplog.records) == 0


def test_nice_console_errors_os_error(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc_info:
        with utils.nice_console_errors():
            raise FileNotFoundError("'./foo.txt' does not exist")

    assert len(caplog.messages) == 1
    record = caplog.records[0]
    assert record.name == 'emsarray.cli.errors.command.FileNotFoundError'
    assert record.message == "'./foo.txt' does not exist"

    exc: SystemExit = exc_info.value
    assert exc.code == 2


def test_nice_console_errors_command_exception(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc_info:
        with utils.nice_console_errors():
            raise CommandException("Could not frobnicate the splines", code=5)

    assert len(caplog.messages) == 1
    record = caplog.records[0]
    assert record.name == 'emsarray.cli.errors.command'
    assert record.message == "Could not frobnicate the splines"

    exc: SystemExit = exc_info.value
    assert exc.code == 5


def test_nice_console_errors_uncaught_exception(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc_info:
        with utils.nice_console_errors():
            1 / 0

    assert len(caplog.messages) == 1
    record = caplog.records[0]
    assert record.name == 'emsarray.cli.errors.uncaught'
    assert record.message == "Uncaught exception: division by zero"

    exc: SystemExit = exc_info.value
    assert exc.code == 3


@pytest.mark.parametrize(
    ['args', 'expected'],
    [
        ([], 1),
        (['-v'], 2),
        (['-vv'], 3),
        (['-q'], 0),
        (['--silent'], 0),
    ],
)
def test_add_verbosity_group(args: list[str], expected: int) -> None:
    parser = argparse.ArgumentParser()
    utils.add_verbosity_group(parser)
    options = parser.parse_args(args)
    assert options.verbosity == expected


def test_bounds_argument() -> None:
    expected = box(1.5, -.2, 3., 4)
    actual = utils.bounds_argument("1.5 , -.2 , 3.,4")

    assert actual.equals(expected)


def test_bounds_argument_invalid() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        utils.bounds_argument("nope")


def test_geometry_argument_bounds() -> None:
    expected = box(1.5, .2, 3., 4)
    actual = utils.geometry_argument("1.5 , .2 , 3.,4")

    assert actual.equals(expected)


def test_geometry_argument_geojson() -> None:
    expected = box(1, 2, 3, 4)
    actual = utils.geometry_argument(
        '{"type": "Polygon", "coordinates": [[[1,2],[3,2],[3,4],[1,4],[1,2]]]}')

    assert actual.equals(expected)


@pytest.mark.parametrize(
    "json_data", [
        # the coordinates array is missing a level of nesting
        {"type": "Polygon", "coordinates": [
            [1, 2], [3, 2], [3, 4], [1, 4], [1, 2]
        ]},
        # Invalid type
        {"type": "nope", "coordinates": [[1, 2]]},
        # Empty
        {},
    ]
)
def test_geometry_argument_bad_geojson(json_data: Any) -> None:
    argument = json.dumps(json_data)
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid geojson string"):
        utils.geometry_argument(argument)


def test_geometry_argument_geojson_file(tmpdir: Path) -> None:
    expected = box(1, 2, 3, 4)

    path = tmpdir / 'polygon.geojson'
    with open(path, 'w') as f:
        geojson.dump(expected, f)
    actual = utils.geometry_argument(str(path))

    assert actual.equals(expected)


@pytest.mark.parametrize("data", [
    "nope",
    json.dumps({"not": "geojson"}),
])
def test_geometry_argument_invalid_geojson_file(tmpdir: Path, data: str) -> None:
    path = tmpdir / 'polygon.geojson'
    with open(path, 'w') as f:
        f.write(data)
    with pytest.raises(argparse.ArgumentTypeError, match="File is not valid geojson"):
        utils.geometry_argument(str(path))
