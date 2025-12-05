import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import emsarray
from emsarray.cli import BaseCommand, CommandException
from emsarray.operations import geometry

logger = logging.getLogger(__name__)


format_writers: dict[str, Callable] = {
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

        parser.add_argument(
            "-g", "--grid-kind", type=str,
            default=None,
            help=(
                "Grid kind to export. Will export the default grid if not specified."
            ),
        )

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

        if options.grid_kind is None:
            grid_kind = dataset.ems.default_grid_kind
        else:
            grid_kind_names = {str(grid_kind): grid_kind for grid_kind in dataset.ems.grid_kinds}
            try:
                grid_kind = grid_kind_names[options.grid_kind]
            except KeyError:
                grid_kind_choices = ", ".join(grid_kind_names.keys())
                raise CommandException(
                    f"Unknown grid kind {options.grid_kind!r}. "
                    f"Valid choices are: {grid_kind_choices}."
                )

        try:
            writer = format_writers[output_format]
        except KeyError:
            raise CommandException(f"Unknown output format {output_format!r}")

        grid = dataset.ems.grids[grid_kind]
        geometries = grid.geometry[grid.mask]
        count = geometries.size

        logger.debug("Grid kind: %s", grid_kind)
        logger.debug("Geometry type: %s", grid.geometry_type.__name__)
        logger.debug("Geometry count: %d", count)
        logger.debug("Exporting geometry as %r", output_format)

        writer(dataset, output_path, grid_kind=grid_kind)
