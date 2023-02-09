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
run the ``./scripts/release.py pre`` script.
This will create a new branch and update the code base ready for a new release.
Push this branch and make a pull request on Github.

.. code-block:: console

   $ ./scripts/release.py pre 1.2.0

Manually run the 'Prerelease checks' workflow for this branch in the 'Actions' tab on Github.
This will check that a conda package can be built from the Python package tarball.

Build and publish packages
==========================

Once the release pull request has been merged
tag the merge commit with the correct version.
The ``scripts/tag-release.py`` command will do the hard work for you:

.. code-block:: shell

   $ ./scripts/release.py tag

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

Once the new version has been released
the code base needs to be prepared so development work can continue:

.. code-block:: shell

   $ ./scripts/release.py post

Push this branch and create a pull request.
Once this pull request has been merged,
the release process is finished!
