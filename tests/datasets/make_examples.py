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
    values[:] = np.arange(nj * ni).reshape((nj, ni))

    dataset.close()


@dataset_maker
def make_ugrid_mesh2d(out: pathlib.Path) -> None:
    nfaces = 200
    radius = 3

    envelope_detail = 12
    envelope = shapely.geometry.Polygon([
        (
            np.cos(i / envelope_detail * 2 * np.pi) * radius * 1.2,
            np.sin(i / envelope_detail * 2 * np.pi) * radius * 1.2,
        )
        for i in range(envelope_detail)
    ])

    # A Fibonacci spiral, with the radii increasing slightly faster than normal
    # to space the faces out.
    phi = (1 + 5 ** 0.5) / 2
    angles = np.arange(nfaces) * 2 * np.pi / (phi ** 2)
    radii = np.linspace(0, 1, nfaces) ** 0.7 * radius
    points = np.stack((
        np.cos(angles) * radii,
        np.sin(angles) * radii,
    ), axis=-1)

    # Make faces from the points by finding the Voronoi diagram of the centres.
    voronoi = shapely.ops.voronoi_diagram(
        shapely.geometry.MultiPoint(points), envelope=envelope)
    faces = [polygon.intersection(envelope) for polygon in voronoi.geoms]

    # Get the unique vertices of the faces
    nodes = np.array(list(set(
        p for polygon in faces
        for p in polygon.exterior.coords
    )))
    # A map between {point: index}
    node_indices = dict((tuple(p), i) for i, p in enumerate(nodes))
    # Number of vertices
    nnodes = len(node_indices)
    # Maximum vertex count for any face
    max_vertex_count = max(len(polygon.exterior.coords) for polygon in faces) - 1
    # A sensible fill value of all nines
    fill_value = 10 ** (int(np.floor(np.log10(nnodes + 1))) + 1) - 1

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
    values[:] = np.arange(nfaces)

    dataset.close()


@dataset_maker
def make_ugrid_mesh2d_one_indexed(out: pathlib.Path) -> None:
    inner_point_count = 8
    nnodes = inner_point_count * 3
    nfaces = nnodes
    nfaces = nnodes

    inner_radius = 1
    outer_radius = 3

    inner_point_x = inner_radius * np.cos(
        np.linspace(0, 2, inner_point_count, endpoint=False) * np.pi)
    inner_point_y = inner_radius * np.sin(
        np.linspace(0, 2, inner_point_count, endpoint=False) * np.pi)

    outer_point_x = outer_radius * np.cos(
        np.linspace(0, 2, inner_point_count * 2, endpoint=False) * np.pi)
    outer_point_y = outer_radius * np.sin(
        np.linspace(0, 2, inner_point_count * 2, endpoint=False) * np.pi)

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
    node_x[:] = np.concatenate([inner_point_x, outer_point_x], axis=None)
    node_y = dataset.createVariable("node_y", "f4", ["node"])
    node_y[:] = np.concatenate([inner_point_y, outer_point_y], axis=None)

    face_node_connectivity = dataset.createVariable(
        "mesh_face_node", "i1", ["face", "max_vertex"])
    face_node_connectivity.start_index = 1

    for i in range(inner_point_count):
        inner = (np.array([0, 1]) + i) % inner_point_count + 1
        outer = (np.array([0, 1, 2]) + i * 2) % (inner_point_count * 2) + 1 + inner_point_count
        face_node_connectivity[i * 3 + 0, :] = [inner[0], outer[1], outer[0]]
        face_node_connectivity[i * 3 + 1, :] = [inner[0], inner[1], outer[1]]
        face_node_connectivity[i * 3 + 2, :] = [inner[1], outer[2], outer[1]]

    values = dataset.createVariable("values", "i1", ["face"])
    values[:] = np.arange(nfaces)

    dataset.close()


def main() -> None:
    here = pathlib.Path(__file__).parent

    make_cfgrid1d(here / 'cfgrid1d.nc')
    make_cfgrid2d(here / 'cfgrid2d.nc')
    make_shoc_standard(here / 'shoc_standard.nc')
    make_ugrid_mesh2d(here / 'ugrid_mesh2d.nc')
    make_ugrid_mesh2d_one_indexed(here / 'ugrid_mesh2d_one_indexed.nc')


if __name__ == '__main__':
    main()
