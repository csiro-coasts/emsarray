"""
Utilities to make writing robust and friendly command line scripts easier.
"""
import argparse
import contextlib
import decimal
import json
import logging.config
import re
import sys
import textwrap
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Protocol

from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from emsarray.conventions._registry import entry_point_conventions

from .exceptions import CommandException

cli_logger = logging.getLogger('emsarray.cli')
error_logger = cli_logger.getChild('errors')
uncaught_exception_logger = error_logger.getChild('uncaught')
command_exception_logger = error_logger.getChild('command')


class MainCallable(Protocol):
    def __call__(
        self,
        argv: Optional[List[str]] = None,
        handle_errors: bool = True,
    ) -> None:
        ...


class NamespaceCallable(Protocol):
    def __call__(self, options: argparse.Namespace) -> None:
        ...


def console_entrypoint(
    add_arguments: Callable[[argparse.ArgumentParser], None]
) -> Callable[[NamespaceCallable], MainCallable]:
    """
    Decorator that turns a function in to a console entrypoint, suitable to use
    as a ``main()`` function. It will automatically set up loggers, make an
    argument parser, add verbosity flags, and handle unexpected errors.

    :func:`.console_entrypoint` takes one argument, a function that sets up an
    :class:`argparse.ArgumentParser` instance.
    :func:`.console_entrypoint` will use this to configure an :class:`~argparse.ArgumentParser`,
    parse the command line flags, and pass the options to the decorated function.

    Example
    -------

    This is a complete example of a command line program that will clip an
    input file using a geometry.

    .. code-block:: python

        #!/usr/bin/env python3

        import emsarray
        import tempfile
        from emsarray.cli.utils import console_entrypoint, geometry_argument
        from pathlib import Path

        def command_line_flags(parser: argparse.ArgumentParser) -> None:
            parser.add_argument('input_file', type=Path)
            parser.add_argument('clip', type=geometry_argument)
            parser.add_argument('output_file', type=Path)

        @console_entrypoint(command_line_flags)
        def main(options: argparse.Namespace) -> None:
            dataset = emsarray.open_dataset(options.input_file)
            with tempfile.TemporaryDirectory() as temp_dir:
                dataset.clip(options.clip, work_dir=Path(temp_dir))
                dataset.ems.to_netcdf(options.output_file)

        if __name__ == '__main__':
            main()

    This example is more-or-less exactly what
    the :func:`.console_entrypoint` decorator does,
    where the decorated function provides the rest of the implementation.

    .. code-block:: python

        @nice_console_errors()
        def main(argv: Optional[List[str]]) -> None:
            parser = argparse.ArgumentParser()
            add_verbosity_group(parser)
            command_line_flags(parser)
            options = parser.parse_args(argv)
            set_verbosity(options.verbosity)

            ...  # Continue on with the rest of the program

    See Also
    --------

    :func:`.nice_console_errors`
    :func:`.set_verbosity`
    :func:`.add_verbosity_group`
    """
    def decorator(
        fn: NamespaceCallable,
    ) -> MainCallable:
        @wraps(fn)
        def wrapper(
            argv: Optional[List[str]] = None,
            handle_errors: bool = True,
        ) -> None:
            parser = argparse.ArgumentParser(
                formatter_class=DoubleNewlineDescriptionFormatter,
                add_help=False)
            logging_group = parser.add_argument_group('logging options')
            logging_group.add_argument(
                '-h', '--help', action='help',
                help="Show this help message and quit")
            add_verbosity_group(logging_group)
            add_arguments(parser)
            options = parser.parse_args(argv)
            set_verbosity(options.verbosity)
            cli_logger.debug("Command line options:")
            for name, value in sorted(vars(options).items(), key=lambda i: i[0]):
                cli_logger.debug("%s = %r", name, value)

            if handle_errors:
                with nice_console_errors():
                    fn(options)
            else:
                fn(options)

        return wrapper
    return decorator


@contextlib.contextmanager
def nice_console_errors() -> Iterator:
    """
    A decorator or context manager that sends uncaught errors to the
    appropriate loggers and then quits.
    """
    try:
        yield
    except OSError as err:
        os_error_logger = command_exception_logger.getChild(type(err).__name__)
        os_error_logger.exception(str(err))
        sys.exit(2)
    except CommandException as err:
        # This is a deliberate error raised to exit the program early.
        # Tracebacks are suppressed.
        command_exception_logger.error(err.message)
        sys.exit(err.code)
    except Exception as err:
        uncaught_exception_logger.exception('Uncaught exception: ' + str(err))
        sys.exit(3)
    except KeyboardInterrupt:
        # Handling KeyboardInterrupt properly is tricky, and full of opinions.
        # This is close enough. It will exit with a non-zero exit code,
        # will allow 'atexit' handlers to run and clean things up,
        # but will not set WEXITSTATUS
        cli_logger.info('Caught KeyboardInterrupt, exiting')
        print("")
        sys.exit(1)


class DoubleNewlineDescriptionFormatter(argparse.HelpFormatter):
    def _fill_text(self, text: str, width: int, indent: str) -> str:
        fill_text = super(DoubleNewlineDescriptionFormatter, self)._fill_text

        return '\n\n'.join(
            fill_text(paragraph, width, indent)
            if not paragraph.startswith(' ')
            else paragraph
            for paragraph in textwrap.dedent(text).split('\n\n')
        )


def add_verbosity_group(parser: argparse._ActionsContainer) -> None:
    """
    Add ``--verbose`` and ``--silent`` mutually exclusive flags to an
    :class:`~argparse.ArgumentParser`.
    """
    parser.set_defaults(verbosity=1)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        '-v', '--verbose', action='count', dest='verbosity',
        help=(
            "Print more progress information to the console. "
            "Repeating this flag increases the verbosity further."
        ))
    verbosity.add_argument(
        '-q', '--silent', action='store_const', const=0, dest='verbosity',
        help="Print nothing except errors to the console.")


def set_verbosity(level: int) -> None:
    """
    Configure a sensible logging set up, with configurable verbosity.
    """
    if level <= 0:
        level_str = 'ERROR'
    if level == 1:
        level_str = 'WARNING'
    elif level == 2:
        level_str = 'INFO'
    elif level >= 3:
        level_str = 'DEBUG'

    # Include logging handlers for all plugins
    entry_point_convention_modules = sorted({
        convention.__module__
        for convention in entry_point_conventions()
        if not convention.__module__.startswith('emsarray.')
    })

    logging.captureWarnings(True)
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'format': '[%(asctime)s %(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'error': {
                'format': '%(message)s',
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': level_str,
            },
            'error': {
                'class': 'logging.StreamHandler',
                'formatter': 'error',
                'level': level_str,
            },
        },
        'loggers': {
            'emsarray': {'handlers': ['console'], 'level': level_str},
            'emsarray.cli.errors': {
                'handlers': ['error'], 'level': level_str, 'propagate': False
            },
            **{
                module: {'handlers': ['console'], 'level': level_str}
                for module in entry_point_convention_modules
            },
            'py.warnings': {'handlers': ['console'], 'level': 'WARNING'},
            '__main__': {'handlers': ['console'], 'level': level_str},
            'shapely': {'handlers': [], 'level': 60},
        },
        'root_logger': {'handlers': ['console'], 'level': 'WARNING'}
    })


NUMBER = r'\d+(?:_\d+)*'
DECIMAL = rf'(-?(?:{NUMBER}|{NUMBER}\.|\.{NUMBER}|{NUMBER}\.{NUMBER}))'
bounds_re = re.compile(r'\s*,\s*'.join([DECIMAL] * 4))


def geometry_argument(argument_string: str) -> BaseGeometry:
    """Try and make some geometry from an argument.
    The following things are tried in order:

    If the argument consists of four comma separated numbers,
    this is converted to a bounding box.
    The numbers are interpreted as (lon min, lat min, lon max, lat max).

    If the argument can be parsed as JSON, it is assumed to be geojson and
    passed to :func:`shapely.geometry.shape`.

    Finally, the argument is interpreted as a file.
    The file extension is used to guess the type of file.
    Currently geojson files are the only supported type

    Example
    -------

    .. code-block:: python

        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('bounds', type=geometry_argument)
        >>> options = parser.parse_args(['1,2,3,4'])
        >>> print(options.bounds)
        POLYGON (( 3 4, 1 4, 1 2, 3 2, 3 4 ))
        >>> options = parser.parse_args(['{"type": "Point", "coordinates": [1, 2]}'])
        >>> print(options.bounds)
        POINT (1 2)
        >>> options = parser.parse_args(['./path/to/polygon.geojson'])
        >>> print(options.bounds)
        POLYGON (( 0 1, 1 0, 2 1, 1 2, 0 1 ))
    """
    bounds_match = bounds_re.match(argument_string)
    if bounds_match is not None:
        try:
            return box(*map(float, bounds_match.groups()))
        except ValueError:
            pass

    try:
        json_value = json.loads(argument_string)
    except (ValueError, KeyError):
        pass
    else:
        try:
            return shape(json_value)
        except Exception as err:
            # `shape` raises lots of errors - IndexError, AttributeError,
            # KeyError. Being more specific than Exception is hard.
            raise argparse.ArgumentTypeError(f"Invalid geojson string: {err}") from err

    geometry_path = Path(argument_string)
    if not geometry_path.exists():
        raise argparse.ArgumentTypeError(
            "Argument can not be parsed as bounds, geojson, "
            "or a path to an existing geofile")

    if geometry_path.suffix in {'.geojson', '.json'}:
        try:
            with geometry_path.open('r') as f:
                return shape(json.load(f))
        except Exception as err:
            raise argparse.ArgumentTypeError(f"File is not valid geojson: {err}") from err

    else:
        raise argparse.ArgumentTypeError("Unsupported file type")


def bounds_argument(bounds_string: str) -> BaseGeometry:
    """
    Parse a comma separated string of (lon_min, lat_min, lon_max, lat_max)
    in to a :class:`shapely.Polygon`. Used as an :mod:`argparse`
    parameter ``type``.

    Example
    -------

    .. code-block:: python

        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('bounds', type=bounds_argument)
        >>> options = parser.parse_args(['1,2,3,4'])
        >>> print(options.bounds)
        POLYGON (( 3 4, 1 4, 1 2, 3 2, 3 4 ))
    """
    match = bounds_re.match(bounds_string)
    if match is not None:
        try:
            return box(*map(float, match.groups()))
        except decimal.DecimalException:
            pass
    raise argparse.ArgumentTypeError("Expecting four comma separated numbers for bounds")
