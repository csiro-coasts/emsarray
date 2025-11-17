#!/bin/bash

set -e

PYTHON_VERSIONS=('3.12' '3.13' '3.14')
HERE="$( cd -- "$( realpath -- "$( dirname -- "$0" )" )" && pwd )"
PROJECT_ROOT="$( dirname "$HERE" )"

PIP_TOOLS='pip-tools!=7.5.0'

cd "$PROJECT_ROOT"

conda_venv_root=$( mktemp -d emsarray-conda-environments.XXXXXXX )
echo "Working in ${conda_venv_root}"

version="${PYTHON_VERSIONS[0]}"
requirements_file="./continuous-integration/requirements-minimum.txt"
echo "Updating $requirements_file"
conda_prefix="${conda_venv_root}/py-min"
conda create \
	--yes --quiet \
	--prefix="${conda_prefix}" \
	--no-default-packages
conda install \
	--yes \
	--prefix="${conda_prefix}" \
	--channel conda-forge \
	"python=${version}" pip
conda run \
	--prefix="${conda_prefix}" \
	pip install "$PIP_TOOLS" packaging requests python-dateutil
conda run \
	--prefix="${conda_prefix}" \
	python3 ./scripts/min_deps_check.py "$requirements_file"
conda env remove --yes --prefix="${conda_prefix}"

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
		--channel conda-forge \
		"python=${version}" \
		"$PIP_TOOLS"
	conda run \
		--prefix="${conda_prefix}" \
		pip-compile \
			--upgrade \
			--extra="testing" \
			--output-file="${requirements_file}" \
			--unsafe-package emsarray \
			--no-allow-unsafe \
			pyproject.toml
	conda env remove --yes --prefix="${conda_prefix}"
done


echo rm -rf "$conda_venv_root"
