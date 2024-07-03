import pathlib
import subprocess
from unittest import mock

import pytest

from emsarray import nco


def test_ncrcat(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_file = tmp_path / 'out.nc'

    check_call = mock.MagicMock(subprocess.check_call)
    monkeypatch.setattr(subprocess, 'check_call', check_call)

    nco.ncrcat(["one.nc", "two.nc"], out_file)
    check_call.assert_called_once_with(
        ['ncrcat', '-hH', '-o', str(out_file), 'one.nc', 'two.nc'],
        stdin=subprocess.DEVNULL,
    )


def test_ncrcat_output_exists(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_file = tmp_path / 'out.nc'
    out_file.touch()

    check_call = mock.MagicMock(subprocess.check_call)
    monkeypatch.setattr(subprocess, 'check_call', check_call)

    with pytest.raises(FileExistsError):
        nco.ncrcat(["one.nc", "two.nc"], out_file)

    check_call.assert_not_called()


def test_ncrcat_no_input_files(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_file = tmp_path / 'out.nc'

    check_call = mock.MagicMock(subprocess.check_call)
    monkeypatch.setattr(subprocess, 'check_call', check_call)

    with pytest.raises(ValueError):
        nco.ncrcat([], out_file)

    check_call.assert_not_called()
    assert not out_file.exists()
