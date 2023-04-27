import argparse
import functools
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Text, TypeVar

import emsarray
from emsarray.cli import BaseCommand, CommandException

T = TypeVar('T')

logger = logging.getLogger(__name__)


def key_value(arg: str, value_type: Callable = str) -> Dict[str, T]:
    try:
        name, value = arg.split("=", 2)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Coordinate / dimension indexes must be given as `name=value` pairs")
    return {name: value_type(value)}


class UpdateDict(argparse.Action):
    def __init__(
        self,
        option_strings: List[str],
        dest: str,
        *,
        value_type: Callable = str,
        default: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if default is None:
            default = {}
        type = functools.partial(key_value, value_type=value_type)
        super().__init__(
            option_strings, dest, default=default, type=type, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[Text] = None,
    ) -> None:
        super().__call__
        holder = getattr(namespace, self.dest, {})
        print(namespace, holder, values, option_string)
        holder.update(values)
        setattr(namespace, self.dest, holder)


class Command(BaseCommand):
    help = "Plot a dataset variable"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input_path", type=Path,
            help="Path to input netCDF4 file")
        parser.add_argument(
            "variable", type=str, nargs='?', default=None,
            help="Name of a variable to plot")
        parser.add_argument(
            "-s", "--sel", action=UpdateDict,
            help="Select a single coordinate value")
        parser.add_argument(
            "-i", "--isel", action=UpdateDict, value_type=int,
            help="Select a single dimension index")

    def handle(self, options: argparse.Namespace) -> None:
        dataset = emsarray.open_dataset(options.input_path)
        convention: emsarray.Convention = dataset.ems

        if options.variable is None:
            logger.info("Plotting %r", str(options.input_path))
            data_array = None
        else:
            variable_name = options.variable
            if variable_name not in dataset.variables:
                raise CommandException(f"Dataset has no variable {variable_name!r}")
            data_array = dataset[variable_name]

            logger.info("Plotting %r of %r", variable_name, str(options.input_path))

            logger.debug("data_array %s", data_array)
            if options.sel:
                data_array = data_array.sel(options.sel)
            if options.isel:
                data_array = data_array.isel(options.isel)

        convention.plot(data_array)
