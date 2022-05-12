"""
Python aliases to ``nco`` utilities.

:mod:`xarray` is great, but can not handle large files in the same way that
``nco`` tools can. This module provides a set of Python functions that call
``nco`` tools, for times when :mod:`xarray` is not sufficient.

All these functions take paths to datasets as arguments, not
:class:`xarray.Dataset` instances.
"""
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence, Union

Pathish = Union[Path, str]


def _check_call(cmd: Sequence[str], stdin: Any = subprocess.DEVNULL, **kwargs: Any) -> None:
    subprocess.check_call(cmd, stdin=stdin, **kwargs)


def ncrcat(
    input_files: Sequence[Pathish],
    output_file: Pathish,
    flags: Optional[str] = None,
    history: bool = False,
) -> None:
    """Concatenates a set of netCDF files together using `ncrcat`."""
    if Path(output_file).exists():
        raise FileExistsError(f"Output file {str(output_file)!r} already exists!")
    if len(input_files) == 0:
        raise ValueError("Input file list is empty")

    cmd = ['ncrcat']
    if not history:
        cmd += ['-hH']
    if flags:
        cmd += flags
    cmd += ['-o', str(output_file)]
    cmd.extend(map(str, input_files))
    _check_call(cmd)
