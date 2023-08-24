import argparse
import logging
from pathlib import Path
from typing import Callable, Dict

import xarray

import emsarray
from emsarray.cli import BaseCommand, CommandException
from emsarray.operations import geometry
from emsarray.types import Pathish

logger = logging.getLogger(__name__)

Writer = Callable[[xarray.Dataset, Pathish], None]
format_writers: Dict[str, Writer] = {
    'geojson': geometry.write_geojson,
    'shapefile': geometry.write_shapefile,
    'wkt': geometry.write_wkt,
    'wkb': geometry.write_wkb,
}


class Command(BaseCommand):
    help = "Export the geometry of a dataset"
    description = """
    Export the geometry of a dataset to a file
    """

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input_path", type=Path,
            metavar='input-dataset',
            help="Path to input netCDF4 file")
        parser.add_argument(
            'output_path', type=Path,
            metavar='output',
            help=("Path to exported geometry"))

        parser.add_argument(
            "-f", "--format", type=str,
            default="auto",
            choices=("auto", "geojson", "wkt", "wkb", "shapefile"),
            help=(
                "Format for exported geometry. "
                "The output format will be guessed using the output extension by default"
            ))

    def guess_format(self, output_path: Path) -> str:
        extension = output_path.suffix
        if extension in {'.json', '.geojson'}:
            return 'geojson'
        if extension == '.wkt':
            return 'wkt'
        if extension == '.wkb':
            return 'wkb'
        if extension == '.shp':
            return 'shapefile'
        raise CommandException(
            f"Could not guess output format from extension {extension!r}")

    def handle(self, options: argparse.Namespace) -> None:
        logger.info("Exporting geometry for %r", str(options.input_path))
        dataset = emsarray.open_dataset(options.input_path)

        output_path: Path = options.output_path
        output_format: str = options.format
        if output_format == 'auto':
            output_format = self.guess_format(output_path)
            logger.debug("Guessed output format as %r", output_format)

        count = dataset.ems.polygons[dataset.ems.mask].size
        logger.debug("Dataset contains %d polygons", count)

        try:
            writer = format_writers[output_format]
        except KeyError:
            raise CommandException(f"Unknown output format {output_format!r}")

        logger.debug("Exporting geometry as %r", output_format)
        writer(dataset, output_path)
