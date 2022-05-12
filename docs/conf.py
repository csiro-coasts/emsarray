# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import importlib
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import pathlib
import sys

project_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / 'src'))


# -- Project information -----------------------------------------------------

project = 'emsarray'
copyright = '2021, CSIRO'
author = 'Coastal Environmental Modelling team, Oceans and Atmosphere, CSIRO'

# The full version, including alpha/beta/rc tags
release = importlib.import_module('emsarray').__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    'xarray.core.dataset.Dataset': ':class:`~xarray.Dataset',
    'xarray.core.dataarray.DataArray': ':class:`~xarray.DataArray',
}


autodoc_typehints = "none"
# autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_default_options = {
    # 'members': True,
    'member-order': 'bysource',
}
autodoc_type_aliases = {
    'IndexType': 'emsarray.formats.IndexType',
}


# Other documentation that we link to
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}
