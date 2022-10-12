import argparse
import contextlib
import logging
import tempfile
from pathlib import Path
from typing import ContextManager

import emsarray
from emsarray.cli import BaseCommand
from emsarray.cli.utils import geometry_argument
from emsarray.types import Pathish

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Clip a dataset to the given geographic bounds"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input_path", type=Path,
            help="Path to input netCDF4 file")
        parser.add_argument(
            'clip_geometry', type=geometry_argument, metavar='clip',
            help=(
                "Shape to clip the dataset to. "
                "Can be a bounding box defined as 'lon_min,lat_min,lon_max,lat_max'; "
                "a WKT string; a geojson string; or a path to a geojson file."
            ))
        parser.add_argument(
            "output_path", type=Path,
            help="Path to output netCDF4 file")

        parser.add_argument(
            "--work_dir", type=Path, default=None,
            help=(
                "Where to put temporary files. If not provided, "
                "the system temporary directory will be used."
            ))

    def handle(self, options: argparse.Namespace) -> None:
        work_context: ContextManager[Pathish]
        if options.work_dir:
            work_context = contextlib.nullcontext(options.work_dir)
        else:
            work_context = tempfile.TemporaryDirectory(prefix="emsarray-clip.")

        logger.info("Clipping %r", str(options.input_path))
        dataset = emsarray.open_dataset(options.input_path)
        with work_context as work_path:
            logger.debug("Using %r as work dir", str(work_path))
            out = dataset.ems.clip(options.clip_geometry, work_dir=work_path)
            logger.debug("Saving to %r", str(options.output_path))
            out.ems.to_netcdf(options.output_path)
        logger.info("Saved to %r", str(options.output_path))
