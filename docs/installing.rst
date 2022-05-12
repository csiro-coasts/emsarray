.. _installing:

==========
Installing
==========

To install ``emsarray``, build a python package and install that using ``pip``.

Building
========

On any computer, run the following commands to build a package:

.. code-block:: shell-session

    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip3 install --upgrade pip build
    $ rm -rf dist/
    $ python3 -m build

Two new files will be created in the ``dist/`` directory.
This is the Python package you can install in other environments.
Use either one of them when installing ``emsarray`` in your chosen environment:

.. code-block:: shell-session

    $ cd /path/to/other-project
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip3 install /path/to/emsarray/dist/emsarray-*-py3-none-any.whl
