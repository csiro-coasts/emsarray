==========================
Frequently asked questions
==========================

What geometry conventions are supported?
========================================

emsarray supports the core geometry conventions
listed in :doc:`/api/conventions/index`.
Additional conventions are :doc:`supported via plugins </developing/conventions>`.
The following plugins are currently available:

* `Spherical multiple-cell geometry <https://github.com/csiro-coasts/emsarray-smc>`_

Why do I get an ``AttributeError`` when accessing ``dataset.ems``?
==================================================================

Check that you have first imported the ``emsarray`` module
before accessing the ``dataset.ems`` attribute.
See :ref:`registering_accessor` for more details.

How should I cite emsarray?
===========================

If you are using emsarray and would like to cite it in an academic publication,
we recommend the following biblatex entry:

.. citation::
   :citation_file: ../CITATION.cff
   :format: biblatex

Or the following APA-like citation:

.. citation::
   :citation_file: ../CITATION.cff
   :format: apa
