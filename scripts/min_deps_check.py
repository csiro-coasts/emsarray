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
import sys
from collections.abc import Iterator

import packaging.requirements
import packaging.version
import requests
from dateutil.relativedelta import relativedelta

IGNORE_DEPS: set[str] = {
    'certifi',
    'pytz',
    'tzdata',
}

POLICY_MONTHS = {"numpy": 18}
POLICY_MONTHS_DEFAULT = 12
POLICY_OVERRIDE: dict[str, packaging.specifiers.Specifier] = {}
errors: list[str] = []


UTC_NOW = datetime.datetime.now(datetime.UTC)


def error(msg: str) -> None:
    global errors
    errors.append(msg)


def warning(msg: str) -> None:
    print("WARNING:", msg)


def parse_requirements(
    fname: str
) -> Iterator[tuple[str, packaging.specifiers.Specifier, packaging.version.Version]]:
    """Load requirements/min-all-deps.yml

    Yield (package name, major version, minor version, patch version)
    """
    for line_number, line in enumerate(open(fname, 'r'), start=1):
        if '#' in line:
            line = line[:line.index('#')]
        line = line.strip()
        if not line:
            continue

        requirement = packaging.requirements.Requirement(line)
        if requirement.name in IGNORE_DEPS:
            continue

        if len(requirement.specifier) != 1:
            error(f"Dependency on line {line_number} not in expected format: {line}")
            continue
        specifier = next(iter(requirement.specifier))

        if specifier.operator != '~=':
            error(f"Specificity for dependency {requirement.name} should be '~='")

        version = packaging.version.parse(specifier.version)
        if version.micro is None:
            warning(
                f"Dependency {requirement.name} does not specify a patch version. "
                f"Update dependency to `{version}.0`."
            )

        yield requirement.name, specifier, version


def zero_patch(version: packaging.version.Version) -> packaging.version.Version:
    return packaging.version.parse(f'{version.major}.{version.minor}.0')


def query_pypi(pkg: str) -> dict[packaging.version.Version, datetime.datetime]:
    """Query the conda repository for a specific package

    Return map of {version: publication date}
    """
    response = requests.get(f'https://pypi.org/pypi/{pkg}/json').json()
    all_release_dates = {
        packaging.version.parse(key): min(
            datetime.datetime.fromisoformat(entry['upload_time_iso_8601'])
            for entry in entries
        )
        for key, entries in response['releases'].items()
        if len(entries) > 0
    }

    return all_release_dates


def process_pkg(
    pkg: str,
    specifier: packaging.specifiers.Specifier,
    version: packaging.version.Version,
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
    release_dates = query_pypi(pkg)

    # Find when this version was published
    try:
        req_published = release_dates[version]
    except KeyError:
        error(f"not found in pypi: {pkg}=={version}")
        req_published = None

    policy_months = POLICY_MONTHS.get(pkg, POLICY_MONTHS_DEFAULT)

    if pkg in POLICY_OVERRIDE:
        policy_specifier = POLICY_OVERRIDE[pkg]
    else:
        # Find the newest minor version with a release before the policy cut off date
        policy_published = UTC_NOW - relativedelta(months=policy_months)
        filtered_versions = [
            version
            for version, published in release_dates.items()
            if published < policy_published
        ]
        max_policy_version = zero_patch(max(filtered_versions, default=version))
        policy_specifier = packaging.specifiers.Specifier(f'~={max_policy_version}')

    policy_version = packaging.version.parse(policy_specifier.version)

    # Find the release date of the policy version
    policy_release_date = min(
        (
            release_date
            for release_version, release_date
            in release_dates.items()
            if release_version in policy_specifier
        ), default=None
    )

    if version in policy_specifier:
        status = "~="
    else:
        if version < policy_version:
            status = '<'
            warning(
                f"Requirement {pkg} {version} was published on {req_published:%Y-%m-%d} "
                f"which is older than the required {policy_months} months of support. "
                f"Minimum policy supported version is {pkg}{policy_specifier}."
            )
        elif version > policy_version:
            status = '> (!)'
            if req_published is None:
                error(
                    f"Package version is newer than policy version. "
                    f"Policy wants {policy_specifier} "
                    f"(at least {policy_months} moths old). "
                    f"Could not determine age of {version}."
                )
            else:
                delta = relativedelta(UTC_NOW, req_published).normalized()
                n_months = delta.years * 12 + delta.months
                error(
                    f"Package is too new: {pkg} {version} was "
                    f"published on {req_published:%Y-%m-%d} "
                    f"which was {n_months} months ago (policy is {policy_months} months). "
                    f"Update requirement to {pkg}{policy_specifier}."
                )

    return (
        pkg,
        str(version),
        req_published.strftime("%Y-%m-%d") if req_published else "-",
        str(policy_version),
        policy_release_date.strftime("%Y-%m-%d") if policy_release_date else "-",
        status,
    )


def main() -> None:
    requirements_file = sys.argv[1]
    rows = [process_pkg(pkg, specifier, version) for pkg, specifier, version in parse_requirements(requirements_file)]

    print()
    print("Package              Required                Status Policy                 ")
    print("-------------------- ----------------------- ------ -----------------------")
    fmt = "{0:20} {1:10} ({2:10}) {5:^6} {3:10} ({4:10})"
    for row in rows:
        print(fmt.format(*row))

    if errors:
        print("\nErrors:")
        print("-------")
        for i, e in enumerate(errors, start=1):
            print(f"{i}. {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
