=======
Testing
=======

``emsarray`` uses `tox`_ to run all the test and linting tools.
Python tests use the `pytest`_ framework,
`mypy`_ is used for type checking,
`flake8`_ and `isort`_ check the code formatting,
and the documentation is build using `sphinx`_

Using tox
=========

The easiest way to install all the test requirements is using Conda:

* Make a new Conda environment,
* install the required packages using the provided environment file,
* install Python and tox.

.. code-block:: shell-session

    $ conda env create --name 'emsarray-tests-py3.12' --file continuous-integration/environment.yaml
    $ conda activate emsarray-tests-py3.12
    $ conda install -c conda-forge python==3.12 tox

Invoke ``tox`` to run all the tests:

.. code-block:: shell-session

    $ tox run -e py312-pytest-latest py312-pytest-pinned lint docs

Note that this should only be used to run tests for the version of Python you
installed in your Conda environment.

Using pytest
============

You can invoke ``pytest`` directly to run just a subset of the tests.
Set up a new Conda environment, then install `emsarray` with the `testing` extra:

.. code-block:: shell-session

   $ conda env create --name 'emsarray-development' --file continuous-integration/environment.yaml
   $ conda activate emsarray-development
   $ conda install -c conda-forge python=3.12 pip
   $ pip install -e .[testing]

Invoke ``pytest`` to run the tests:

.. code-block:: shell-session

    $ pytest

.. _tox: https://tox.wiki/
.. _pytest: https://pytest.org/
.. _mypy: https://github.com/python/mypy
.. _flake8: https://flake8.pycqa.org/en/latest/
.. _isort: https://pycqa.github.io/isort/
.. _sphinx: https://www.sphinx-doc.org/en/master/
