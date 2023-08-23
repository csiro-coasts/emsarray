"""
``emsarray`` provides both a command line interface,
and a set of tools to make writing your own command line scripts easier.
"""

import argparse
import importlib
import pkgutil
from typing import Iterable, Type

import emsarray

from . import commands
from .command import BaseCommand
from .exceptions import CommandException
from .utils import console_entrypoint

__all__ = ['main', 'CommandException', 'BaseCommand', 'console_entrypoint']


def command_line_flags(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(func=lambda x: print(parser.format_help()))
    parser.add_argument(
        '-V', '--version', action='version', version=f'%(prog)s {emsarray.__version__}')

    subparsers = parser.add_subparsers(title="operations", metavar='OPERATION')
    for command_cls in _find_all_commands():
        command_cls().add_parser(subparsers)


@console_entrypoint(command_line_flags)
def main(options: argparse.Namespace) -> None:
    """
    The main entry point for :mod:`emsarray` as a command line utility.

    See Also
    --------
    :ref:`cli`
    """
    options.func(options)


def _find_all_commands() -> Iterable[Type[BaseCommand]]:
    for moduleinfo in pkgutil.iter_modules(commands.__path__):
        if moduleinfo.name.startswith('_'):
            continue
        command_path = commands.__name__ + '.' + moduleinfo.name
        command_module = importlib.import_module(command_path)
        yield command_module.Command
