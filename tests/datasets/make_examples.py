"""
Make example datasets for the tests. This script deliberately does not use
emsarray or xarray to construct the datasets. The data associated with the
datasets are meaningless.
"""

import datetime
import functools
import pathlib
from collections.abc import Callable

import netCDF4
import numpy
import shapely.geometry
import shapely.ops


def dataset_maker(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(out: pathlib.Path) -> None:
        print(f"Making {out.name}...")
        if out.exists():
            out.unlink()
        fn(out)

    return wrapper


@dataset_maker
def make_times(out: pathlib.Path) -> None:
    """
    This dataset contains nothing but a time coordinate.
    It is used for basic testing of the time coordinate detection.
    """
    dataset = netCDF4.Dataset(str(out), "w", format="NETCDF4")
    dataset.createDimension("record", None)

    epoc = datetime.date(1990, 1, 1)
    one_day = datetime.timedelta(days=1)
    start_days_since_epoc = (datetime.date(2023, 8, 17) - epoc) // one_day
    time = dataset.createVariable("time", "f4", ["record"])
    time[:] = numpy.arange(start_days_since_epoc, start_days_since_epoc + 10)
    time.units = f"days since {epoc:%Y-%m-%d %H:%M:%S Z}"

    dataset.close()


@dataset_maker
def make_cfgrid1d(out: pathlib.Path) -> None:
    nlat, nlon = 21, 21
    shape = (nlat, nlon)
    size = numpy.prod(shape)

    dataset = netCDF4.Dataset(str(out), "w", format="NETCDF4")
    dataset.createDimension("lat", nlat)
    dataset.createDimension("lon", nlon)

    lat = dataset.createVariable(
        "lat", "f4", ["lat"],
        zlib=True)
    lat[:] = numpy.linspace(0, 10, nlat)
    lat.standard_name = 'latitude'
    lat.axis = 'Y'

    lon = dataset.createVariable(
        "lon", "f4", ["lon"],
        zlib=True)
    lon[:] = numpy.linspace(0, 10, nlon)
    lon.standard_name = 'longitude'
    lon.axis = 'X'

    values = dataset.createVariable(
        "values", "i4", ["lat", "lon"],
        zlib=True)
    values[:] = numpy.arange(size).reshape(shape)

    dataset.close()


@dataset_maker
def make_cfgrid2d(out: pathlib.Path) -> None:
    nj, ni = 15, 21
    jj, ii = numpy.mgrid[0:nj, 0:ni]

    dataset = netCDF4.Dataset(str(out), "w", format="NETCDF4")
    dataset.createDimension("i", ni)
    dataset.createDimension("j", nj)

    theta, radius = ii / ni, (jj / 3) + 2

    lat = dataset.createVariable("lat", "f4", ["j", "i"])
    lat[:] = numpy.sin(theta) * radius
    lat[-2:, :3] = numpy.nan
    lat.standard_name = 'latitude'

    lon = dataset.createVariable("lon", "f4", ["j", "i"])
    lon[:] = numpy.cos(theta) * radius
    lon[-2:, :3] = numpy.nan
    lon.standard_name = 'longitude'

    values = dataset.createVariable("values", "i4", ["j", "i"])
    values[:] = numpy.arange(nj * ni).reshape((nj, ni))

    dataset.close()


@dataset_maker
def make_shoc_standard(out: pathlib.Path) -> None:
    # Very similar to the cfgrid2d, with multiple interleaved grids
    nj, ni = 15, 21
    jj, ii = numpy.mgrid[0:nj + 1, 0:ni + 1]

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
    xx = numpy.cos(theta) * radius
    yy = numpy.sin(theta) * radius

    face_y = dataset.createVariable("y_centre", "f4", ["face_j", "face_i"])
    face_y[:] = (yy[:-1, :-1] + yy[:-1, 1:] + yy[1:, 1:] + yy[1:, :-1]) / 4

    face_x = dataset.createVariable("x_centre", "f4", ["face_j", "face_i"])
    face_x[:] = (xx[:-1, :-1] + xx[:-1, 1:] + xx[1:, 1:] + xx[1:, :-1]) / 4

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
    values[:] = numpy.arange(nj * ni).reshape((nj, ni))

    dataset.close()


@dataset_maker
def make_ugrid_mesh2d(out: pathlib.Path) -> None:
    nfaces = 200
    radius = 3

    envelope_detail = 12
    envelope = shapely.geometry.Polygon([
        (
            numpy.cos(i / envelope_detail * 2 * numpy.pi) * radius * 1.2,
            numpy.sin(i / envelope_detail * 2 * numpy.pi) * radius * 1.2,
        )
        for i in range(envelope_detail)
    ])

    # A Fibonacci spiral, with the radii increasing slightly faster than normal
    # to space the faces out.
    phi = (1 + 5 ** 0.5) / 2
    angles = numpy.arange(nfaces) * 2 * numpy.pi / (phi ** 2)
    radii = numpy.linspace(0, 1, nfaces) ** 0.7 * radius
    points = numpy.stack((
        numpy.cos(angles) * radii,
        numpy.sin(angles) * radii,
    ), axis=-1)

    # Make faces from the points by finding the Voronoi diagram of the centres.
    voronoi = shapely.ops.voronoi_diagram(
        shapely.geometry.MultiPoint(points), envelope=envelope)
    faces = [polygon.intersection(envelope) for polygon in voronoi.geoms]

    # Get the unique vertices of the faces
    nodes = numpy.array(list({
        p for polygon in faces
        for p in polygon.exterior.coords
    }))
    # A map between {point: index}
    node_indices = {tuple(p): i for i, p in enumerate(nodes)}
    # Number of vertices
    nnodes = len(node_indices)
    # Maximum vertex count for any face
    max_vertex_count = max(len(polygon.exterior.coords) for polygon in faces) - 1
    # A sensible fill value of all nines
    fill_value = 10 ** (int(numpy.floor(numpy.log10(nnodes + 1))) + 1) - 1

    dataset = netCDF4.Dataset(str(out), "w", format="NETCDF4")
    dataset.Conventions = 'UGRID'

    mesh = dataset.createVariable("Mesh2D", "i4", [])
    mesh.cf_role = "mesh_topology"
    mesh.topology_dimension = 2
    mesh.node_coordinates = "node_x node_y"
    mesh.face_node_connectivity = "mesh_face_node"

    dataset.createDimension("node", nnodes)
    dataset.createDimension("face", nfaces)
    dataset.createDimension("max_vertex", max_vertex_count)

    node_x = dataset.createVariable("node_x", "f4", ["node"])
    node_x[:] = nodes[:, 0]
    node_y = dataset.createVariable("node_y", "f4", ["node"])
    node_y[:] = nodes[:, 1]
    face_node_connectivity = dataset.createVariable(
        "mesh_face_node", "i4", ["face", "max_vertex"],
        fill_value=fill_value)
    for iface, polygon in enumerate(faces):
        coords = polygon.exterior.coords
        face_node = [node_indices[p] for p in coords[:-1]]
        face_node_connectivity[iface, :len(face_node)] = face_node

    values = dataset.createVariable("values", "i4", ["face"])
    values[:] = numpy.arange(nfaces)

    dataset.close()


@dataset_maker
def make_ugrid_mesh2d_one_indexed(out: pathlib.Path) -> None:
    inner_point_count = 8
    nnodes = inner_point_count * 3
    nfaces = nnodes
    nfaces = nnodes

    inner_radius = 1
    outer_radius = 3

    inner_point_x = inner_radius * numpy.cos(
        numpy.linspace(0, 2, inner_point_count, endpoint=False) * numpy.pi)
    inner_point_y = inner_radius * numpy.sin(
        numpy.linspace(0, 2, inner_point_count, endpoint=False) * numpy.pi)

    outer_point_x = outer_radius * numpy.cos(
        numpy.linspace(0, 2, inner_point_count * 2, endpoint=False) * numpy.pi)
    outer_point_y = outer_radius * numpy.sin(
        numpy.linspace(0, 2, inner_point_count * 2, endpoint=False) * numpy.pi)

    dataset = netCDF4.Dataset(str(out), "w", format="NETCDF4")
    dataset.Conventions = 'UGRID'

    mesh = dataset.createVariable("Mesh2D", "i4", [])
    mesh.cf_role = "mesh_topology"
    mesh.topology_dimension = 2
    mesh.node_coordinates = "node_x node_y"
    mesh.face_node_connectivity = "mesh_face_node"

    dataset.createDimension("node", nnodes)
    dataset.createDimension("face", nfaces)
    dataset.createDimension("max_vertex", 3)

    node_x = dataset.createVariable("node_x", "f4", ["node"])
    node_x[:] = numpy.concatenate([inner_point_x, outer_point_x], axis=None)
    node_y = dataset.createVariable("node_y", "f4", ["node"])
    node_y[:] = numpy.concatenate([inner_point_y, outer_point_y], axis=None)

    face_node_connectivity = dataset.createVariable(
        "mesh_face_node", "i1", ["face", "max_vertex"])
    face_node_connectivity.start_index = 1

    for i in range(inner_point_count):
        inner = (numpy.array([0, 1]) + i) % inner_point_count + 1
        outer = (numpy.array([0, 1, 2]) + i * 2) % (inner_point_count * 2) + 1 + inner_point_count
        face_node_connectivity[i * 3 + 0, :] = [inner[0], outer[1], outer[0]]
        face_node_connectivity[i * 3 + 1, :] = [inner[0], inner[1], outer[1]]
        face_node_connectivity[i * 3 + 2, :] = [inner[1], outer[2], outer[1]]

    values = dataset.createVariable("values", "i1", ["face"])
    values[:] = numpy.arange(nfaces)

    dataset.close()


def main() -> None:
    here = pathlib.Path(__file__).parent

    make_times(here / 'times.nc')
    make_cfgrid1d(here / 'cfgrid1d.nc')
    make_cfgrid2d(here / 'cfgrid2d.nc')
    make_shoc_standard(here / 'shoc_standard.nc')
    make_ugrid_mesh2d(here / 'ugrid_mesh2d.nc')
    make_ugrid_mesh2d_one_indexed(here / 'ugrid_mesh2d_one_indexed.nc')


if __name__ == '__main__':
    main()
