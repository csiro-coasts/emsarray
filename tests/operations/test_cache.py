import hashlib
import pathlib

import pytest
import xarray

import emsarray
import emsarray.operations.cache

# Sha1
attr_hash_lon = "4151c4ac3e0e913dd2c5520c0b2c33e6f20eb004"
str_hash = '67a0425c3bcb8a47ccc39615e59ab489d0c4b6a1'
int_hash = '7b08e025e311c3dfcf5179b67c0fdc08e73de261'
attr_hash_lat = "2cb433979fc2d9c3884eea8569dd6a44406950f3"
cache_key_hash_cf1d_sha1 = "2b006999273225ed70d4810357b6a06e6bebe9a6"
cache_key_hash_multifile_cf2d_sha1 = "ea2d2e6131f1e499f622e83ed4fc2415649def06"
cache_key_hash_multifile_ugrid_mesh2d_sha1 = "1d72e01b159135208324ae9a643166f85aecba27"

# Blake2b
cache_key_hash_cf1d = "1a3226072f08441ee79f727b0775709209ff2965299539c898ecc401cf17e23f"
cache_key_hash_cf2d = "c8d2671af35e96a08c6e33f616bc460ba3957bfa3ffc4529cfd84378e9de4c2f"
cache_key_hash_shoc_standard = "18788cd57e4aff55dc5c0918513942bc6849d7b90e53a09bb77c4598fa3daf29"
cache_key_hash_ugrid_mesh2d = "9e7c0ded2bfcc6843131e09303b8bec53e6edfbb18115e2c50e07adce29e8702"
cache_key_hash_ugrid_mesh2d_one_indexed = "045be69d5ef70b20f26dc9a14e6cea8a11fe3cc9a3c4c4b303aed6f22206ac7a"


@pytest.fixture(autouse=True)
def override_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The cache key depends on the emsarray version.
    This fixture overrides the version with a constant value
    so we don't have to regenerate the test cache keys every time we release a new version.
    """
    monkeypatch.setattr(emsarray, '__version__', '1.0.0')


def test_hash_attributes(datasets: pathlib.Path):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    m_lon = hashlib.sha1()

    emsarray.operations.cache.hash_attributes(m_lon, dataset.variables['lon'].attrs)

    result_hash_lon = m_lon.hexdigest()

    assert result_hash_lon is not None

    assert result_hash_lon == attr_hash_lon


def test_hash_string():
    hash = hashlib.sha1()

    emsarray.operations.cache.hash_string(hash, "1234")

    result_hash = hash.hexdigest()

    assert result_hash is not None

    assert result_hash == str_hash


def test_hash_int():
    hash = hashlib.sha1()

    emsarray.operations.cache.hash_int(hash, 1234)

    result_hash = hash.hexdigest()

    assert result_hash is not None

    assert result_hash == int_hash


def test_hash_int_overflow():

    emsarray.operations.cache.hash_int(hashlib.sha1(), 2 ** 31 - 5)

    with pytest.raises(OverflowError):
        emsarray.operations.cache.hash_int(hashlib.sha1(), 2 ** 31)


def test_cache_key_cfgrid1d(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_cf1d


def test_cache_key_cfgrid2d(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid2d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_cf2d


def test_cache_key_shoc_standard(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'shoc_standard.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_shoc_standard


def test_cache_key_ugrid_mesh2d(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'ugrid_mesh2d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_ugrid_mesh2d


def test_cache_key_ugrid_mesh2d_one_indexed(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'ugrid_mesh2d_one_indexed.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_ugrid_mesh2d_one_indexed


def test_cache_key_is_deterministic(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    result_cache_key_cf_1 = emsarray.operations.cache.make_cache_key(dataset_cf)
    result_cache_key_cf_2 = emsarray.operations.cache.make_cache_key(dataset_cf)

    assert result_cache_key_cf_1 == result_cache_key_cf_2

    assert result_cache_key_cf_1 == cache_key_hash_cf1d


def test_hash_attributes_is_deterministic(datasets: pathlib.Path):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    m_lon_1 = hashlib.sha1()
    m_lon_2 = hashlib.sha1()

    emsarray.operations.cache.hash_attributes(m_lon_1, dataset.variables['lon'].attrs)
    emsarray.operations.cache.hash_attributes(m_lon_2, dataset.variables['lon'].attrs)

    result_hash_lon_1 = m_lon_1.hexdigest()
    result_hash_lon_2 = m_lon_2.hexdigest()

    assert result_hash_lon_1 == result_hash_lon_2

    assert result_hash_lon_1 == attr_hash_lon


def test_cache_key_gives_unique_keys(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')
    dataset_ugrid = emsarray.open_dataset(datasets / 'ugrid_mesh2d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf)
    result_cache_key_ugrid = emsarray.operations.cache.make_cache_key(dataset_ugrid)

    assert result_cache_key_cf is not None
    assert result_cache_key_ugrid is not None

    assert result_cache_key_cf != result_cache_key_ugrid

    assert result_cache_key_cf == cache_key_hash_cf1d
    assert result_cache_key_ugrid == cache_key_hash_ugrid_mesh2d


def test_hash_attributes_gives_unique_keys(datasets: pathlib.Path):
    dataset = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    m_lon = hashlib.sha1()
    m_lat = hashlib.sha1()

    emsarray.operations.cache.hash_attributes(m_lon, dataset.variables['lon'].attrs)
    emsarray.operations.cache.hash_attributes(m_lat, dataset.variables['lat'].attrs)

    result_hash_lon = m_lon.hexdigest()
    result_hash_lat = m_lat.hexdigest()

    assert result_hash_lon is not None
    assert result_hash_lat is not None

    assert result_hash_lon != result_hash_lat

    assert result_hash_lon == attr_hash_lon
    assert result_hash_lat == attr_hash_lat


def test_cache_key_cfgrid1d_sha1(datasets: pathlib.Path):
    dataset_cf = emsarray.open_dataset(datasets / 'cfgrid1d.nc')

    result_cache_key_cf = emsarray.operations.cache.make_cache_key(dataset_cf, hashlib.sha1())

    assert result_cache_key_cf is not None

    assert result_cache_key_cf == cache_key_hash_cf1d_sha1


def test_cache_key_with_multifile_dataset_ugrid_mesh2d(datasets: pathlib.Path):

    ugrid_path1 = datasets / 'multiple_dataset/ugrid_mesh2d/ugrid_mesh2d_2024-01-01.nc'
    ugrid_path2 = datasets / 'multiple_dataset/ugrid_mesh2d/ugrid_mesh2d_2024-01-02.nc'

    dataset_paths = [ugrid_path1, ugrid_path2]

    multifile_dataset = xarray.open_mfdataset(dataset_paths, data_vars=['values'])

    multifile_ds_hash = hashlib.sha1()

    multifile_dataset.ems.hash_geometry(multifile_ds_hash)

    multifile_ds_digest = multifile_ds_hash.hexdigest()

    assert multifile_ds_digest == cache_key_hash_multifile_ugrid_mesh2d_sha1


def test_cache_key_with_multifile_dataset_cfgrid2d(datasets: pathlib.Path):

    cfgrid_path1 = datasets / 'multiple_dataset/cfgrid2d/cfgrid2d_2024-01-01.nc'
    cfgrid_path2 = datasets / 'multiple_dataset/cfgrid2d/cfgrid2d_2024-01-02.nc'

    dataset_paths = [cfgrid_path1, cfgrid_path2]

    multifile_dataset = xarray.open_mfdataset(dataset_paths, data_vars=['values'])

    multifile_ds_hash = hashlib.sha1()

    multifile_dataset.ems.hash_geometry(multifile_ds_hash)

    multifile_ds_digest = multifile_ds_hash.hexdigest()

    assert multifile_ds_digest == cache_key_hash_multifile_cf2d_sha1
