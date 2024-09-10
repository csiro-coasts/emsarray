=============================
Next release (in development)
=============================

* Fix invalid geometry being generated for 'river' cells
  in CFGrid2D datasets with no cell bounds (:pr:`154`).
* Improved speed of triangulation for convex polygons
  (:pr:`151`).
* Check all polygons in a dataset are valid as part of generating them.
  This will slow down opening new datasets slightly,
  but the trade off is worth the added security
  after the invalid polygons found in :pr:`154`
  (:pr:`156`).
