import abc
import argparse
from typing import Optional

from emsarray.cli import utils


class BaseCommand(abc.ABC):
    """
    Base class for writing :mod:`emsarray` command line tools.
    Subclasses of this can be added to the :mod:`emsarray` entry point :func:`.main`.
    """

    @property
    def name(self) -> str:
        """
        The name of this subcommand.
        Users can invoke this subcommand by running ``emsarray <name>``.
        Defaults to the name of the module,
        with underscores replaced with hyphens.
        """
        module_name = type(self).__module__.rsplit('.', 1)[-1]
        return module_name.replace('_', '-')

    #: A short description of what this subcommand does,
    #: shown as part of the usage message for the base command.
    help: Optional[str] = None

    #: A longer description of what this subcommand does,
    #: shown as part of the usage message for this subcommand.
    description: Optional[str] = None

    def add_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            name=self.name,
            help=self.help,
            description=self.description,
            formatter_class=utils.DoubleNewlineDescriptionFormatter,
        )
        parser.set_defaults(func=self.handle)
        self.add_arguments(parser)

    @abc.abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Configure the :class:`argparse.ArgumentParser` for this command,
        adding any flags and options required.
        """
        ...

    @abc.abstractmethod
    def handle(self, options: argparse.Namespace) -> None:
        """
        Run this command. ``options`` will be the parsed command line flags,
        as configured in :meth:`.add_arguments`.
        """
        ...
