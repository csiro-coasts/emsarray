==============================================
Dependency versions for continuous integration
==============================================

In order to make CI reproducible we pin the installed Python dependencies.
These dependencies should be updated every few months in order to stay relevant.

CI tests are also run against the latest of all dependencies in a separate job.
This job is allowed to fail without breaking the build.
It acts as a canary so we can see new breakages as they happen
without needing to fix issues from new package versions in unrelated pull requests.

Supported Python versions and dependency versions follows `SPEC-0000 <https://scientific-python.org/specs/spec-0000/>`_.

Updating supported Python versions
==================================

To add a new supported version of Python the following changes must be made:

* Update the pinned dependencies:
  add a new version to the list in ``scripts/update_pinned_dependencies.sh``, and
  rebuild the pinned dependencies by running the script.
* Update ``.github/workflows/ci.yaml``:
  set ``env.python-version`` to the new version,
  add the new version to the ``test`` job ``python-version`` matrix variable, and
  update the version associated with the ``dependencies: "latest"`` matrix job.
* Update ``tox.ini``:
  add the new version to the ``envlist`` for pinned dependencies, and
  update the version used in the latest dependencies.
* Add a release note.

To remove support for an old version of Python the following changes must be made:

* Update the pinned dependencies:
  remove the old version from the list in ``scripts/update_pinned_dependencies.sh``,
  remove the old ``continuous-integration/requirements-X.YY.txt`` file, and
  rebuild the pinned dependencies by running the script.
* Update ``.github/workflows/ci.yaml``:
  remove the version from the ``test`` job ``python-version`` matrix variable, and
  update the version associated with the ``dependencies: "minimum"`` matrix job.
* Update ``tox.ini``:
  remove the old version from the ``envlist`` for pinned dependencies, and
  update the version used in the minimum dependencies.
* Update ``pyproject.toml``:
  update the ``[project] requires-python`` field, and
  update the ``[tool.mypy] python_version`` field.
* Add a release note.

Updating pinned dependencies
============================

To update the list of pinned dependencies used for CI
run the ``scripts/update_pinned_dependencies.sh`` script.
This will create an isolated conda prefix for each of the supported Python versions,
install `pip-tools <https://github.com/jazzband/pip-tools/>`_,
and create a fresh requirements document.
Commit the changes to the pinned requirements,
run the test suite against all supported Python versions, and
fix any new issues that appear.
