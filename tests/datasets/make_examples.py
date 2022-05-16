"""
Make example datasets for the tests. This script deliberately does not use
emsarray or xarray to construct the datasets. The data associated with the
datasets are meaningless.
"""

import functools
import pathlib
from typing import Callable

import netCDF4
import numpy as np


def dataset_maker(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(out: pathlib.Path) -> None:
        print(f"Making {out.name}...")
        if out.exists():
            out.unlink()
        fn(out)

    return wrapper


@dataset_maker
def make_cfgrid1d(out: pathlib.Path) -> None:
    nlat, nlon = 21, 21
    shape = (nlat, nlon)
    size = np.prod(shape)

    dataset = netCDF4.Dataset(str(out), "w", format="NETCDF4")
    dataset.createDimension("lat", nlat)
    dataset.createDimension("lon", nlon)

    lat = dataset.createVariable(
        "lat", "f4", ["lat"],
        zlib=True)
    lat[:] = np.linspace(0, 10, nlat)
    lat.standard_name = 'latitude'
    lat.axis = 'Y'

    lon = dataset.createVariable(
        "lon", "f4", ["lon"],
        zlib=True)
    lon[:] = np.linspace(0, 10, nlon)
    lon.standard_name = 'longitude'
    lon.axis = 'X'

    values = dataset.createVariable(
        "values", "i4", ["lat", "lon"],
        zlib=True)
    values[:] = np.arange(size).reshape(shape)

    dataset.close()


def main() -> None:
    here = pathlib.Path(__file__).parent

    make_cfgrid1d(here / 'cfgrid1d.nc')


if __name__ == '__main__':
    main()
