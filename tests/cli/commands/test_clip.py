import json
import pathlib

import emsarray
from emsarray.cli import main
from emsarray.conventions import UGrid


def test_clip(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    in_path = datasets / 'ugrid_mesh2d.nc'
    geometry = json.dumps({
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
        ],
    })
    out_path = tmp_path / 'out.nc'

    main(['clip', str(in_path), geometry, str(out_path)])

    assert out_path.exists()
    clipped = emsarray.open_dataset(out_path)
    assert isinstance(clipped.ems, UGrid)
