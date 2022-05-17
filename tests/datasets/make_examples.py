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


@dataset_maker
def make_shoc_standard(out: pathlib.Path) -> None:
    # Very similar to the cfgrid2d, with multiple interleaved grids
    nj, ni = 15, 21
    jj, ii = np.mgrid[0:nj + 1, 0:ni + 1]

    dataset = netCDF4.Dataset(str(out), "w", format="NETCDF4")
    dataset.ems_version = 'EMS version X.Y.Z'

    dataset.createDimension("node_i", ni + 1)
    dataset.createDimension("node_j", nj + 1)
    dataset.createDimension("left_i", ni + 1)
    dataset.createDimension("left_j", nj)
    dataset.createDimension("back_i", ni)
    dataset.createDimension("back_j", nj + 1)
    dataset.createDimension("face_i", ni)
    dataset.createDimension("face_j", nj)

    theta, radius = ii / (ni + 1), (jj / 3) + 2
    xx = np.cos(theta) * radius
    yy = np.sin(theta) * radius

    face_y = dataset.createVariable("y_centre", "f4", ["face_j", "face_i"])
    face_y[:] = (yy[:-1, :-1] + yy[:-1, 1:] + yy[1:, 1:] + yy[1:, :-1]) / 2

    face_x = dataset.createVariable("x_centre", "f4", ["face_j", "face_i"])
    face_x[:] = (xx[:-1, :-1] + xx[:-1, 1:] + xx[1:, 1:] + xx[1:, :-1]) / 2

    left_y = dataset.createVariable("y_left", "f4", ["left_j", "left_i"])
    left_y[:] = (yy[:-1, :] + yy[1:, :]) / 2

    left_x = dataset.createVariable("x_left", "f4", ["left_j", "left_i"])
    left_x[:] = (xx[:-1, :] + xx[1:, :]) / 2

    back_y = dataset.createVariable("y_back", "f4", ["back_j", "back_i"])
    back_y[:] = (yy[:, :-1] + yy[:, 1:]) / 2

    back_x = dataset.createVariable("x_back", "f4", ["back_j", "back_i"])
    back_x[:] = (xx[:, :-1] + xx[:, 1:]) / 2

    node_y = dataset.createVariable("y_grid", "f4", ["node_j", "node_i"])
    node_y[:] = yy

    node_x = dataset.createVariable("x_grid", "f4", ["node_j", "node_i"])
    node_x[:] = xx

    values = dataset.createVariable("values", "i4", ["face_j", "face_i"])
    values[:] = np.arange(nj * ni).reshape((nj, ni))

    dataset.close()


def main() -> None:
    here = pathlib.Path(__file__).parent

    make_cfgrid1d(here / 'cfgrid1d.nc')
    make_cfgrid2d(here / 'cfgrid2d.nc')
    make_shoc_standard(here / 'shoc_standard.nc')


if __name__ == '__main__':
    main()
