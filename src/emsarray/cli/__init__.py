"""
:mod:`emsarray` provides both a command line interface,
and a set of tools to make writing your own command line scripts easier.
"""

import argparse

import emsarray

from ._operation import Operation
from .clip import Clip
from .exceptions import CommandException
from .extract_points import ExtractPoints
from .utils import console_entrypoint

__all__ = ['main', 'CommandException', 'Operation']


def command_line_flags(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(func=lambda x: print(parser.format_help()))
    parser.add_argument(
        '-V', '--version', action='version', version=f'%(prog)s {emsarray.__version__}')

    subparsers = parser.add_subparsers(title="Operations")
    Clip().add_parser(subparsers)
    ExtractPoints().add_parser(subparsers)


@console_entrypoint(command_line_flags)
def main(options: argparse.Namespace) -> None:
    """
    The main entry point for :mod:`emsarray` as a command line utility.

    See also
    --------
    :ref:`cli`
    """
    options.func(options)
