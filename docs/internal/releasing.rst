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

Once the release pull request has been merged
tag the merge commit with the correct version.
The ``scripts/tag-release.py`` command will do the hard work for you:

.. code-block:: shell

   $ ./scripts/tag-release.py

Fork the [emsarray-feedstock](https://github.com/conda-forge/emsarray-feedstock) repository
and make a ``release/v1.2.0`` branch.
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
