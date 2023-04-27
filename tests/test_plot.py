import numpy as np
import pytest
from shapely.geometry import Polygon

from emsarray.plot import polygons_to_collection


@pytest.mark.matplotlib
def test_polygons_to_collection():
    polygons = [
        Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)])
        for i in range(10)
    ]
    data = np.random.random(10) * 10
    patch_collection = polygons_to_collection(
        polygons, array=data, cmap='autumn', clim=(0, 10))

    # Check that the polygons came through
    assert len(patch_collection.get_paths()) == 10

    # Check that keyword arguments were passed through
    assert patch_collection.get_cmap().name == 'autumn'
    np.testing.assert_equal(patch_collection.get_array(), data)
