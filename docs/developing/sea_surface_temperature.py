#!/usr/bin/env python3
import argparse
import logging
import pathlib

import emsarray
import xarray as xr
from emsarray.cli import console_entrypoint
from emsarray.cli.utils import geometry_argument
from shapely.geometry.base import BaseGeometry

# Log progress messages using `logger.info(...)`,
# debug messages using `logger.debug(...)`,
# and warning messages using `logger.warn(...)` or `logger.error(...)`.
# Print regular output using `print()`.
logger = logging.getLogger(__name__)


def command_line_flags(parser: argparse.ArgumentParser) -> None:
    """
    Specify all command line arguments here. These can be datasets to
    work with, or names of variables, date ranges, polygons and other
    geometries....

    See https://docs.python.org/3/library/argparse.html
    """
    parser.add_argument(
        "dataset", type=pathlib.Path,
        help="Path to a dataset")
    parser.add_argument(
        "geometry", type=geometry_argument,
        help="The region in which to calculate mean sea surface temperature")
    parser.add_argument(
        "-t", "--temperature-variable",
        type=str, default="temp", metavar="temp",
        help=(
            "The name of the temperature variable in the dataset. "
            "Defaults to 'temp'"
        ))


@console_entrypoint(command_line_flags)
def main(options: argparse.Namespace) -> None:
    """
    This function is where the script will start. `options` will have an
    attribute for each command line flag defined. Access them as
    `options.dataset` or `options.geometry` etc.
    """

    # Log what we are about to do
    logger.info(
        "Calculating mean sea surface temperature for %r in region %s",
        str(options.dataset), options.geometry.wkt)

    # Open the dataset and do our calculations
    dataset = emsarray.open_dataset(options.dataset)
    mean_sea_surface_temperature = calculate_mean_sea_surface_temperature(
        dataset=dataset,
        temperature=dataset[options.temperature_variable],
        geometry=options.geometry)

    # Print the output
    print(mean_sea_surface_temperature)


def calculate_mean_sea_surface_temperature(
    dataset: xr.Dataset,
    temperature: xr.DataArray,
    geometry: BaseGeometry,
) -> float:
    """
    Calculate the mean sea surface temperature inside a region.
    """
    ...  # Implementation left as an exercise
    return 18.0


# This `if` needs to be the last thing in the script
if __name__ == '__main__':
    main()
