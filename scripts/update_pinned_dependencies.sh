#!/bin/bash

set -e

PYTHON_VERSIONS=('3.10' '3.11' '3.12')
HERE="$( cd -- "$( realpath -- "$( dirname -- "$0" )" )" && pwd )"
PROJECT_ROOT="$( dirname "$HERE" )"

cd "$PROJECT_ROOT"

conda_venv_root=$( mktemp -d emsarray-conda-environments.XXXXXXX )
echo "Working in ${conda_venv_root}"

for version in "${PYTHON_VERSIONS[@]}" ; do
	requirements_file="./continuous-integration/requirements-${version}.txt"
	echo "Updating $requirements_file"

	conda_prefix="${conda_venv_root}/py${version}"
	conda create \
		--yes --quiet \
		--prefix="${conda_prefix}" \
		--no-default-packages
	conda install \
		--yes \
		--prefix="${conda_prefix}" \
		"python=${version}" \
		pip-tools
	conda run \
		--prefix="${conda_prefix}" \
		pip-compile \
			--upgrade \
			--extra="testing" \
			--output-file="${requirements_file}" \
			setup.cfg
	conda env remove --yes --prefix="${conda_prefix}"
done

echo rm -rf "$conda_venv_root"
