#!/usr/bin/env bash

# Build a conda package from the locally build python package.
# This is used in CI to test that the conda package can be built successfully.
#
# Prerequisites:
# - Python package builds at ./dist/emsarray-X.Y.Z.tar.gz
# - conda-forge/emsarray-feedstock cloned at ./emsarray-feedstock

set -euo pipefail

HERE="$( cd -- "$( dirname -- "$( realpath -- "$0" )" )" && pwd )"
PROJECT="$( dirname -- "$HERE" )"

dist_path="$PROJECT/dist"
feedstock_path="$PROJECT/emsarray-feedstock"

files_to_clean=()

function main() {
	trap cleanup EXIT

	local package_path="$( get_package_path )"
	local package_name="$( basename -- "$package_path")"
	local package_version="$( get_package_version "$package_name" )"
	local python_min_version="$( get_python_min_version )"

	tmp_dir="$( mktemp -d --tmpdir "emsarray-conda-build.XXXXXXX" )"
	files_to_clean+=( "$tmp_dir" )
	cd "$tmp_dir"

	cp -R "$feedstock_path/recipe" "$tmp_dir/recipe"
	recipe_path="$tmp_dir/recipe/meta.yaml"
	update_recipe "$recipe_path" "$package_path" "$package_version" "$python_min_version"

	cat "$recipe_path"

	conda build \
		--override-channels \
		--channel conda-forge \
		"$tmp_dir"
}

function get_package_path() {
	find "$dist_path" -name 'emsarray-*.tar.gz' -print -quit
}

function get_package_version() {
	local package_name="$1"
	local package_version=$(
		echo "$package_name" \
		| sed 's/^emsarray-\(.*\)\.tar\.gz/\1/'
	)
	if [[ "$package_name" == "$package_version" ]] ; then
		echo "Could not extract version from package name!"
		exit 1
	fi
	echo "$package_version"
}

function get_python_min_version() {
	local pyproject_path="$PROJECT/pyproject.toml"
	local python_min_version="$( cat "$pyproject_path" \
		| grep "requires-python" \
		| sed 's!requires-python = "[>=]*\(.*\)".*$!\1!' )"

	if [[ -z "$python_min_version" ]] ; then
		echo "Could not find requires-python line in pyproject.toml"
		exit 1
	fi

	echo "$python_min_version"
}

function update_recipe() {
	local recipe_path="$1"
	local package_path="$2"
	local package_version="$3"
	local python_min_version="$4"

	sed \
		-e 's!{% set version = .* %}!{% set version = "'"$package_version"'" %}!' \
		-e 's!{% set python_min = .* %}!{% set python_min = "'"$python_min_version"'" %}!' \
		-e 's!url: https://pypi.org/.*!url: "file://'"$package_path"'"!' \
		-e '/sha256:/d' \
		-i "$recipe_path"
}

function cleanup() {
	rm -rf "${files_to_clean[@]}"
}

main
