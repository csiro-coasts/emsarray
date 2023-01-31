===================================
Releasing a new version of emsarray
===================================

Releasing a new version is a multi-step process
that can broadly be summarised as:

* Prepare the codebase for release
* Build and publish packages
* Update records and publish notices
* Prepare the codebase for further development

This example will work through the steps for releasing version 1.2.0,
after the release of version 1.1.0.

Preparing the codebase
======================

When it is time to release a new version
make a branch named ``release/1.2.0``.

Update the version in ``setup.cfg``.

Move ``docs/releases/development.rst`` to ``docs/releases/1.2.0.rst``,
and update the title to ``1.2.0``.
Update ``docs/releases/index.rst``
and add a reference to ``docs/releases/1.2.0.rst`` to the toctree.
Ensure the list of changes is up to date by referring to the merged pull requests.

Commit these changes and make a pull request on Github.
Review these changes to ensure everything is correct and all the tests pass.
Merge the pull request.

Build and publish packages
==========================

Fetch the latest ``emsarray`` commits including the freshly merged pull request.
Tag the merge commit from the release pull request.
Note that the tag name includes a ``v`` prefix.

.. code-block:: shell

   $ git tag v1.2.0 <sha-of-merge-commit>
   $ git push origin v1.2.0
   $ git checkout v1.2.0

Release the new version to PyPI:

.. code-block:: shell

   $ rm -rf ./build ./venv
   $ python3 -m venv venv
   $ source venv/bin/activate
   $ pip install build twine
   $ python3 -m build
   $ twine upload dist/*

Fork the [emsarray-feedstock](https://github.com/conda-forge/emsarray-feedstock) repository
and make a ``release/1.2.0`` branch.
Update the version in ``recipe/meta.yaml``.
Ensure the minimum dependencies in ``recipe/meta.yaml`` are correct by comparing with ``setup.cfg``.
Commit these changes and make a pull request.
Review the changes and merge the pull request once everything looks good and the automated builds pass.
This will automatically publish version 1.2.0 to conda-forge.

Update records and publish notices
==================================

[Make a new release on Github](https://github.com/csiro-coasts/emsarray/releases/new).
Select the 1.2.0 tag you just created.
Copy the release notes from ``docs/releases/1.2.0.rst``.

Publish a new version to the [CSIRO Data Access Portal](https://data.csiro.au/collection/csiro:57587v1).
Update the version number and add the PyPI packages and source tarball for this release.

Prepare the codebase for further development
============================================

Make a new file ``docs/releases/development.rst`` with the content

.. code-block:: rst

   =============================
   Next release (in development)
   =============================

   * ...

Update ``docs/releases/index.rst`` to include a reference to this document.
Commit and push this change.
