============
emsarray.cli
============

.. automodule:: emsarray.cli
    :members: main

.. module:: emsarray.cli.utils

Writing command line scripts
----------------------------

A typical Python command line script follows the pattern:

.. code-block:: python

    #!/usr/bin/env python3
    import argparse

    from some_library import frobnicate


    def main():
        # Configure the parser
        parser = argparse.ArgumentParser()
        parser.add_argument('spline', type=int)

        # Parse the command line arguments
        options = parser.parse_args()

        # Do the things that need doing
        frobinate(options.spline)


    if __name__ == '__main__':
        main()

This module provides a few helpers that make this process much easier.

.. autofunction:: emsarray.cli.utils.console_entrypoint
.. autofunction:: emsarray.cli.utils.nice_console_errors
.. autofunction:: emsarray.cli.utils.add_verbosity_group
.. autofunction:: emsarray.cli.utils.set_verbosity


Command line flags
------------------

The following functions aid in converting command line flags
in to useful Python values.

.. autofunction:: emsarray.cli.utils.geometry_argument
.. autofunction:: emsarray.cli.utils.bounds_argument

Module content
--------------

.. autoexception:: emsarray.cli.CommandException
.. autoclass:: emsarray.cli.Operation
