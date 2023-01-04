.. _scripts:

====================
Command line scripts
====================

``emsarray`` provides a set of tools to make it easy
to write your own command line scripts that interact with ``emsarray``,
while promoting robustness and best practice.
These scripts can be used to automate any repetitive process using ``emsarray``.

Build from the following template to get started.
A full copy of this scriptcan be downloaded
for easier reading:
:download:`sea_surface_temperature.py`.
Run this script as `python3 sea_surface_temperature.py --help` to see it in action.

.. literalinclude:: ./sea_surface_temperature.py

This script will automatically gain the following features:

- Command line flags are clearly defined near the top of the file,
- It will generate useful output when run with ``--help``
  or when provided invalid arguments,
- The logging output level can be set using ``-v``,
- Errors are handled gracefully.

