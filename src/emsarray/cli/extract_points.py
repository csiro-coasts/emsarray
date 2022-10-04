import argparse
import contextlib
import logging
import tempfile
from pathlib import Path
from typing import ContextManager

import pandas as pd

import emsarray
from emsarray.operations import point_extraction
from emsarray.utils import to_netcdf_with_fixes

from ._operation import Operation

logger = logging.getLogger(__name__)


class ExtractPoints(Operation):
    name = 'extract-points'
    description = "Extract points from a dataset"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input_path", type=Path,
            help="Path to input netCDF4 file")
        parser.add_argument(
            'points', type=Path, metavar='csv',
            help=(
                "Path to a CSV file with the points to extract"
            ))
        parser.add_argument(
            "output_path", type=Path,
            help="Path to output netCDF4 file")

        parser.add_argument(
            "-c", "--coordinate-columns", type=str, nargs=2,
            default=("lon", "lat"),
            help=(
                "Names of the longitude and latitude columns in the CSV file. "
                "Defaults to 'lon' and 'lat'."
            ))

        parser.add_argument(
            "-d", "--point-dimension", type=str,
            default="point",
            help=(
                "Name of the new dimension to index the point data"
            ))

    def handle(self, options: argparse.Namespace) -> None:
        logger.info("Extracting points from %r", str(options.input_path))
        dataset = emsarray.open_dataset(options.input_path)
        dataframe = pd.read_csv(options.points)

        point_data = point_extraction.extract_dataframe(
            dataset, dataframe, options.coordinate_columns,
            point_dimension=options.point_dimension)
        try:
            time_name = dataset.ems.get_time_name()
        except KeyError:
            time_name = None

        to_netcdf_with_fixes(
            point_data, options.output_path, time_variable=time_name)
