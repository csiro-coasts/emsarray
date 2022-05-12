### Notes

* Need to add show masking flags, outside and LAND
  - or different tool?
* Need flag to double-check dataarray-geometry mashup
  - throw error
* Maybe make the fileformats class based instead
* matplotlib backend could be an issue, need to experiment
* see about passing in axis param
* https://stackoverflow.com/questions/30030328/correct-placement-of-colorbar-relative-to-geo-axes-cartopy
  - for colour bar placement
* Maybe see about modifying dataArray at open_dataset time ...?
* Standard name for record dimension? Yes, good idea, should be able to overwrite
  - same for depth?
  - on the other hand maybe not, maybe just keep the format for actual data handy instead
* Add geojson method
* Add API for data values only
* Add time/depth selections
  - need a way to keep python objects and GUI in sync
* parllel=True in open_mfdataset
* Add coordinates to DataArrays instead of the whole object
* Implement Abstract classes/methods
* Make sure it works with all of our formats, including OCMAPS, ACCESS, SCHISM, SWAN etc ...
* Will need a manual to add specific formats
* Generalise classes/methods as much as possible

#### Next steps:
* Add get record dim
* Add plot method
* Add isel (record, k), sel (time, z) methods
* Add axis parameter
* Bottom/surface and interpolation
* quiver plots
* edgecolors
* ipywidgets
* check out mplcursors
* How should we treat depth layers? Nan's or separate geometries?

