=============
Release notes
=============

0.2.0
=====

* Added support for :doc:`additional formats via plugins <../developing/formats>`
  (:pr:`11`).
* Added support for one-based indexing in UGRID datasets
  (:pr:`14`).
* Buffering around clip regions in :meth:`.Format.make_clip_mask` is now optional
  (:issue:`12`, :pr:`20`).
* Removed dependency on SciPy, added missing dependency to conda package.

0.1.0
=====

* Initial public release
