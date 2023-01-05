import pathlib

import netCDF4

from emsarray import Convention

from .conventions.test_shoc_standard import make_dataset


def test_ems_accessor():
    dataset = make_dataset(j_size=10, i_size=20)
    assert isinstance(dataset.ems, Convention)


def test_to_netcdf(tmp_path: pathlib.Path):
    """
    Test that ``dataset.ems.to_netcdf()`` saves a netCDF4 file with some
    various fixes applied:

    * Time units corrected to be in a EMS-compatible format
    * _FillValue only set when needed

    This needs to be a proper ems dataset so that the accessor works.
    """
    xr_dataset = make_dataset(j_size=10, i_size=10)

    # There are a few problems we work around. Lets assert that they are still
    # problems before we check whether we fixed them.
    xr_dataset.to_netcdf(tmp_path / "bad.nc")
    with netCDF4.Dataset(tmp_path / "bad.nc", "r") as nc_dataset:
        # Formatted in a way that EMS can not parse
        time_units = nc_dataset.variables["t"].getncattr("units")
        assert time_units == "days since 1990-01-01T00:00:00+10:00"

        # The CF conventions state that coordinate variables should not have
        # needless _FillValues set, as all coordinates should be defined.
        # Some of our coordinates are not all defined - x_grid, y_grid for
        # datasets with holes being one example - but z_grid is fully defined
        # and should not have a _FillValue
        assert "_FillValue" in nc_dataset.variables["z_grid"].ncattrs()

    # Save it using dataset.ems.to_netcdf(...)
    xr_dataset.ems.to_netcdf(tmp_path / "good.nc")
    with netCDF4.Dataset(tmp_path / "good.nc", "r+") as nc_dataset:
        # The time units should now be in an EMS-compatible format
        time_units = nc_dataset.variables["t"].getncattr("units")
        assert time_units == 'days since 1990-01-01 00:00:00 +10:00'

        # _FillValue should not be set automatically for all variables
        assert "_FillValue" not in nc_dataset.variables["z_grid"].ncattrs()
