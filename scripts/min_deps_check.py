#!/usr/bin/env python
"""
Fetch from conda database all available versions of the emsarray dependencies and their
publication date. Compare it against continuous-integration/min-deps.yaml to verify the
policy on obsolete dependencies is being followed. Print a pretty report :)

Based heavily on `min_deps_check.py` from xarray
but rewritten to pull requirements from a Python requirements.txt and available versions from PyPI.

Needs the following extra deps installed:

    $ pip3 install packaging requests python-dateutil

See also
========
https://github.com/pydata/xarray/blob/v2024.06.0/ci/min_deps_check.py
"""
import datetime
import itertools
import re
import sys
from collections.abc import Iterator

import packaging.requirements
import packaging.version
import requests
from dateutil.relativedelta import relativedelta

CHANNELS = ["conda-forge", "defaults"]
IGNORE_DEPS = {
    "pytest",
    "pytest-cov",
    "pytest-mpl",
    "mypy",
    "pandas-stubs",
    "types-pytz",
    "flake8",
    "isort",
    "tzdata",
    "pytz",
}

POLICY_MONTHS = {"python": 30, "numpy": 18}
POLICY_MONTHS_DEFAULT = 12
POLICY_OVERRIDE: dict[str, tuple[int, int]] = {}
errors: list[str] = []


UTC_NOW = datetime.datetime.now(datetime.UTC)


def error(msg: str) -> None:
    global errors
    errors.append(msg)
    print("ERROR:", msg)


def warning(msg: str) -> None:
    print("WARNING:", msg)



SPECIFIER_RE = re.compile(r'~=(\d+)\.(\d+)\.(\d+)')

def parse_requirements(fname: str) -> Iterator[tuple[str, int, int, int]]:
    """Load requirements/min-all-deps.yml

    Yield (package name, major version, minor version, patch version)
    """
    for line in open(fname, 'r'):
        if '#' in line:
            line = line[:line.index('#')]
        line = line.strip()
        if not line:
            continue

        requirement = packaging.requirements.Requirement(line)
        if requirement.name in IGNORE_DEPS:
            continue

        specifier_match = SPECIFIER_RE.match(str(requirement.specifier))
        if specifier_match is None:
            error(f"dependency should be specified as {requirement.name}~=major.minor.patch")
            continue

        try:
            major = int(specifier_match[1])
            minor = int(specifier_match[2])
            patch = int(specifier_match[3])
        except ValueError:
            raise ValueError(f"non-numerical version: {requirement}")

        yield (requirement.name, major, minor, patch)



def extract_major_minor(version_string: str) -> tuple[int, int]:
    version = packaging.version.Version(version_string)
    return version.major, version.minor


def query_pypi(pkg: str) -> dict[packaging.version.Version, datetime.datetime]:
    """Query the conda repository for a specific package

    Return map of {version: publication date}
    """
    response = requests.get(f'https://pypi.org/pypi/{pkg}/json').json()
    all_release_dates = sorted([
        (extract_major_minor(key), min(
            datetime.datetime.fromisoformat(entry['upload_time_iso_8601'])
            for entry in entries
        ))
        for key, entries in response['releases'].items()
        if len(entries) > 0
    ])

    major_minor_release_date = {
        major_minor: min(e[1] for e in entries)
        for major_minor, entries in itertools.groupby(all_release_dates, key=lambda e: e[0])
    }

    return major_minor_release_date


def process_pkg(
    pkg: str, req_major: int, req_minor: int, req_patch: int | None
) -> tuple[str, str, str, str, str, str]:
    """Compare package version from requirements file to available versions in conda.
    Return row to build pandas dataframe:

    - package name
    - major.minor.[patch] version in requirements file
    - publication date of version in requirements file (YYYY-MM-DD)
    - major.minor version suggested by policy
    - publication date of version suggested by policy (YYYY-MM-DD)
    - status ("<", "=", "> (!)")
    """
    print(f"Analyzing {pkg}...")
    versions = query_pypi(pkg)

    try:
        req_published = versions[req_major, req_minor]
    except KeyError:
        error(f"not found in pypi: {pkg}~={req_major}.{req_minor}.{0}")
        req_published = None

    policy_months = POLICY_MONTHS.get(pkg, POLICY_MONTHS_DEFAULT)
    policy_published = UTC_NOW - relativedelta(months=policy_months)

    filtered_versions = [
        version
        for version, published in versions.items()
        if published < policy_published
    ]
    policy_major, policy_minor = max(filtered_versions, default=(req_major, req_minor))

    try:
        policy_major, policy_minor = POLICY_OVERRIDE[pkg]
    except KeyError:
        pass
    policy_published_actual = versions[policy_major, policy_minor]

    if (req_major, req_minor) < (policy_major, policy_minor):
        status = "<"
    elif (req_major, req_minor) > (policy_major, policy_minor):
        status = "> (!)"
        if req_published is None:
            warning(
                f"Package version is newer than policy version. "
                f"Policy wants {policy_major}.{policy_minor} "
                f"(at least {policy_months} moths old). "
                f"Could not determine age of {req_major}.{req_minor}."
            )
        else:
            delta = relativedelta(UTC_NOW, req_published).normalized()
            n_months = delta.years * 12 + delta.months
            warning(
                f"Package is too new: {pkg}={req_major}.{req_minor} was "
                f"published on {req_published:%Y-%m-%d} "
                f"which was {n_months} months ago (policy is {policy_months} months)"
            )
    else:
        status = "="

    return (
        pkg,
        fmt_version(req_major, req_minor, req_patch),
        req_published.strftime("%Y-%m-%d") if req_published else "-",
        fmt_version(policy_major, policy_minor),
        policy_published_actual.strftime("%Y-%m-%d"),
        status,
    )


def fmt_version(major: int, minor: int, patch: int | None = None) -> str:
    if patch is None:
        return f"{major}.{minor}"
    else:
        return f"{major}.{minor}.{patch}"


def main() -> None:
    requirements_file = sys.argv[1]
    rows = [
        process_pkg(pkg, major, minor, patch)
        for pkg, major, minor, patch in parse_requirements(requirements_file)
    ]

    print()
    print("Package              Required                Policy                  Status")
    print("-------------------- ----------------------- ----------------------- ------")
    fmt = "{:20} {:10} ({:10}) {:10} ({:10}) {}"
    for row in rows:
        print(fmt.format(*row))

    if errors:
        print("\nErrors:")
        print("-------")
        for i, e in enumerate(errors):
            print(f"{i+1}. {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
