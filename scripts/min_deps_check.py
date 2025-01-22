#!/usr/bin/env python
"""
Fetch from PyPI all available versions of the emsarray dependencies and their
publication date. Compare it against continuous-integration/requirements-minimum.txt
to verify the policy on obsolete dependencies is being followed.
Update the pinned dependencies using `pip-compile`.

Based heavily on `min_deps_check.py` from xarray
but rewritten to pull requirements from a Python requirements.txt and available versions from PyPI.

Needs the following extra deps installed:

    $ pip3 install packaging requests python-dateutil pip-tools

This is automatically run as part of the `scripts/update_pinned_dependencies.sh` script.

See also
========
https://github.com/pydata/xarray/blob/v2024.06.0/ci/min_deps_check.py
"""
import dataclasses
import datetime
import enum
import itertools
import shlex
import subprocess
import sys
from collections.abc import Iterator

import packaging.requirements
import packaging.version
import requests
from dateutil.relativedelta import relativedelta


class VersionStatus(enum.StrEnum):
    older = '<'
    current = '~='
    newer = '>'


@dataclasses.dataclass
class Dependency:
    package_name: str
    requirements_version: packaging.version.Version
    requirements_date: datetime.date | None
    policy_version: packaging.version.Version
    policy_date: datetime.date | None
    status: VersionStatus


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
    filename: str
) -> Iterator[tuple[str, packaging.specifiers.Specifier, packaging.version.Version]]:
    """
    Parse a requirements file, yield (package name, specifier, version) for each requirement.
    """
    for line_number, line in enumerate(open(filename), start=1):
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
) -> Dependency:
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
    policy_date = min(
        (
            release_date
            for release_version, release_date
            in release_dates.items()
            if release_version in policy_specifier
        ), default=None
    )

    if version in policy_specifier:
        status = VersionStatus.current
    else:
        if version < policy_version:
            status = VersionStatus.older
            warning(
                f"Requirement {pkg} {version} was published on {req_published:%Y-%m-%d} "
                f"which is older than the required {policy_months} months of support. "
                f"Minimum policy supported version is {pkg}{policy_specifier}."
            )
        elif version > policy_version:
            status = VersionStatus.newer
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

    return Dependency(
        package_name=pkg,
        requirements_version=version,
        requirements_date=req_published,
        policy_version=policy_version,
        policy_date=policy_date,
        status=status,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} continuous-integration/requirements-minimum.txt")
        sys.exit(1)

    requirements_file = sys.argv[1]
    dependencies = [
        process_pkg(pkg, specifier, version)
        for pkg, specifier, version in parse_requirements(requirements_file)
    ]

    print()
    print("Package              Required                Status Policy                 ")
    print("-------------------- ----------------------- ------ -----------------------")
    fmt = "{0:20} {1:10} ({2:10}) {5:^6} {3:10} ({4:10})"
    for d in dependencies:
        requirements_date = (
            d.requirements_date.strftime("%Y-%m-%d")
            if d.requirements_date is not None
            else "-"
        )
        policy_date = (
            d.policy_date.strftime("%Y-%m-%d")
            if d.policy_date is not None
            else "-"
        )
        print(
            f"{d.package_name:20} {d.requirements_version!s:10} ({requirements_date:10}) "
            f"{d.status:^6} {d.policy_version!s:10} ({policy_date:10})"
        )

    upgrade_args = list(itertools.chain.from_iterable(
        ['--upgrade-package', f'{d.package_name}~={d.policy_version}']
        for d in dependencies
        if d.status is VersionStatus.older
    ))
    maintain_args = list(itertools.chain.from_iterable(
        ['--upgrade-package', f'{d.package_name}~={d.requirements_version}']
        for d in dependencies
        if d.status in {VersionStatus.current, VersionStatus.newer}
    ))
    if upgrade_args:
        ignored_args = list(itertools.chain.from_iterable(
            ['--unsafe-package', ignored]
            for ignored in IGNORE_DEPS
        ))
        cmd = [
            'pip-compile',
            '--quiet',
            '--extra', 'complete',
            '--strip-extras',
            '--unsafe-package', 'emsarray',
            '--no-allow-unsafe',
            # '--no-header',
            # '--no-annotate',
            '--output-file', requirements_file,
        ] + upgrade_args + maintain_args + ignored_args + [
            'pyproject.toml',
        ]
        print('$', shlex.join(cmd))
        subprocess.check_call(cmd)
        cmd = [
            'sed',
            '-i',
            's/==/~=/',
            requirements_file,
        ]
        print('$', shlex.join(cmd))
        subprocess.check_call(cmd)

    if errors:
        print("\nErrors:")
        print("-------")
        for i, e in enumerate(errors, start=1):
            print(f"{i}. {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
