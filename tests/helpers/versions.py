import importlib.metadata

import pytest
from packaging.requirements import Requirement


def skip_versions(*requirements: str):
    """
    Skips a test function if any of the version specifiers match.
    """
    invalid_versions = []
    for requirement in map(Requirement, requirements):
        assert not requirement.extras
        assert requirement.url is None
        assert requirement.marker is None

        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            # The package is not installed, so an invalid version isn't installed
            continue

        if version in requirement.specifier:
            invalid_versions.append(
                f'{requirement.name}=={version} matches skipped version specifier {requirement}')

    return pytest.mark.skipif(len(invalid_versions) > 0, reason='\n'.join(invalid_versions))


def only_versions(*requirements: str):
    """
    Runs a test function only if all of the version specifiers match.
    """
    invalid_versions = []
    for requirement in map(Requirement, requirements):
        assert not requirement.extras
        assert requirement.url is None
        assert requirement.marker is None

        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            # The package is not installed, so a required version is not installed
            invalid_versions.append(f'{requirement.name} is not installed')
            continue

        if version not in requirement.specifier:
            invalid_versions.append(
                f'{requirement.name}=={version} does not satisfy {requirement}')

    return pytest.mark.skipif(len(invalid_versions) > 0, reason='\n'.join(invalid_versions))
