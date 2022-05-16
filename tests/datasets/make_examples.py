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


@dataset_maker
def make_cfgrid2d(out: pathlib.Path) -> None:
    nj, ni = 15, 21
    jj, ii = np.mgrid[0:nj, 0:ni]

    dataset = netCDF4.Dataset(str(out), "w", format="NETCDF4")
    dataset.createDimension("i", ni)
    dataset.createDimension("j", nj)

    theta, radius = ii / ni, (jj / 3) + 2

    lat = dataset.createVariable("lat", "f4", ["j", "i"])
    lat[:] = np.sin(theta) * radius
    lat[-2:, :3] = np.nan
    lat.standard_name = 'latitude'

    lon = dataset.createVariable("lon", "f4", ["j", "i"])
    lon[:] = np.cos(theta) * radius
    lon[-2:, :3] = np.nan
    lon.standard_name = 'longitude'

    values = dataset.createVariable("values", "i4", ["j", "i"])
    values[:] = np.arange(nj * ni).reshape((nj, ni))

    dataset.close()


def main() -> None:
    here = pathlib.Path(__file__).parent

    make_cfgrid1d(here / 'cfgrid1d.nc')
    make_cfgrid2d(here / 'cfgrid2d.nc')


if __name__ == '__main__':
    main()
