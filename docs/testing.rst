=======
Testing
=======

``emsarray`` uses `tox`_ to run all the test and linting tools.
Python tests use the `pytest`_ framework,
`mypy`_ is used for type checking,
`flake8`_ and `isort`_ check the code formatting,
and the documentation is build using `sphinx`_

Setup
=====

To install all the testing tools,
install ``emsarray`` in to a virtual environment with the ``[testing]`` extra.

.. code-block:: shell-session

    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip3 install --upgrade pip
    $ pip3 install -r requirements.txt -e .[testing]

Running all the tests
=====================

Invoke ``tox`` to run all the tests:

.. code-block:: shell-session

    $ tox

Running Python tests
====================

You can invoke ``pytest`` directly to run just a subset of the tests.
For example, to only run the tests in ``tests/operations.py``:

.. code-block:: shell-session

    $ pytest tests/test_operations.py

.. _tox: https://tox.wiki/
.. _pytest: https://pytest.org/
.. _mypy: https://github.com/python/mypy
.. _flake8: https://flake8.pycqa.org/en/latest/
.. _isort: https://pycqa.github.io/isort/
.. _sphinx: https://www.sphinx-doc.org/en/master/
