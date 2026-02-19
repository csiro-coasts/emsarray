import dataclasses
from collections.abc import Iterable
from functools import cached_property
from typing import Any, cast

import numpy
import shapely
import xarray
from cartopy import crs
from matplotlib.axes import Axes
from matplotlib.typing import ColorType

from emsarray.conventions import Convention, Grid
from emsarray.exceptions import NoSuchCoordinateError
from emsarray.types import DataArrayOrName
from emsarray.utils import move_dimensions_to_end, name_to_data_array

from . import artists


# Useful for calculating distances in a AzimuthalEquidistant projection
# centred on some point:
#
#     az = crs.AzimuthalEquidistant(p1.x, p1.y)
#     distance = az.project_geometry(p2).distance(ORIGIN)
ORIGIN = shapely.Point(0, 0)


@dataclasses.dataclass
class TransectPoint:
    """
    A TransectPoint holds information about each vertex along a transect path.
    """
    #: The original point, in the CRS of the line string / dataset.
    point: shapely.Point

    #: An AzimuthalEquidistant CRS centred on this point.
    crs: crs.AzimuthalEquidistant

    #: The distance in metres of this point along the line.
    distance_metres: float

    #: The projected distance along the line of this point.
    #: This is normalised to [0, 1].
    #: The actual value is meaningless but can be used to find
    #: the closest vertex on the line string for any other projected point.
    distance_normalised: float


@dataclasses.dataclass
class TransectSegment:
    """
    A TransectSegment holds information about each intersecting segment of the
    transect path and the dataset cells.
    """
    #: The point where the transect path first intersects this dataset cell
    start_point: shapely.Point
    #: The point where the transect exits this dataset cell
    end_point: shapely.Point
    #: The entire intersection between the transect path and this dataset cell
    intersection: shapely.LineString
    #: The distance along the line in metres to the :attr:`.start_point`
    start_distance: float
    #: The distance along the line in metres to the :attr:`.end_point`
    end_distance: float
    #: The linear index of this dataset cell
    linear_index: int
    #: The polygon of the dataset cell
    polygon: shapely.Polygon


class Transect:
    """
    """
    #: The dataset to plot a transect through
    dataset: xarray.Dataset

    #: The transect path to plot
    line: shapely.LineString

    #: The dataset grid to transect.
    grid: Grid

    def __init__(
        self,
        dataset: xarray.Dataset,
        line: shapely.LineString,
        *,
        grid: Grid | None = None,
    ):
        self.dataset = dataset
        self.convention = cast(Convention, dataset.ems)
        self.line = line
        if grid is None:
            grid = self.convention.default_grid
        self.grid = grid

    @cached_property
    def intersection_bounds(
        self,
    ) -> numpy.ndarray:
        """
        A numpy array of shape `(len(segments), 2)`
        indicating the distance to the start and end of each intersection segment.
        This is a shortcut to :attr:`TransectSegment.start_distance`
        and :attr:`~TransectSegment.end_distance` from :attr:`Transect.segments`.
        """
        return numpy.fromiter(
            (
                [segment.start_distance, segment.end_distance]
                for segment in self.segments
            ),
            # Be explicit here, to handle the case when len(self.segments) == 0.
            # This happens when the transect line does not intersect the dataset.
            # This will result in an empty transect plot.
            count=len(self.segments),
            dtype=numpy.dtype((float, 2)),
        )

    @cached_property
    def linear_indexes(self) -> numpy.ndarray:
        """
        A numpy array of length `len(segments)`
        of the linear indexes of each intersecting polygon, in order.
        This is a shortcut to :attr:`TransectSegment.linear_index`
        from :attr:`Transect.segments`.
        """
        return numpy.fromiter(
            (segment.linear_index for segment in self.segments),
            count=len(self.segments),
            dtype=numpy.dtype(int),
        )

    @cached_property
    def holes(self) -> numpy.ndarray:
        """
        An array with the index of any discontinuities in the transect segments.
        For transect paths that are entirely within the dataset geometry this will be empty.
        For paths that pass in and out of the dataset geometry
        this will be the index of the segment just after the discontinuity.
        Two segments are not contiguous if `segment[n].end_distance != segment[n+1].start_distance`
        """
        bounds = self.intersection_bounds
        return numpy.flatnonzero(bounds[:-1, 1] != bounds[1:, 0]) + 1

    def _crs_for_point(
        self,
        point: shapely.Point,
        globe: crs.Globe | None = None,
    ) -> crs.Projection:
        return crs.AzimuthalEquidistant(
            central_longitude=point.x, central_latitude=point.y, globe=globe)

    @cached_property
    def points(
        self,
    ) -> list[TransectPoint]:
        """
        A list of :class:`TransectPoints <TransectPoint>`,
        one for each point in the transect :attr:`.line`.
        """
        data_crs = self.convention.data_crs
        globe = data_crs.globe

        # Make the TransectPoint for the first point by hand.
        point = shapely.Point(self.line.coords[0])
        points = [TransectPoint(
            point=point,
            crs=self._crs_for_point(point, globe),
            distance_metres=0,
            distance_normalised=0,
        )]

        # Make a TransectPoint for each subsequent point along the line.
        for point in map(shapely.Point, self.line.coords[1:]):
            previous = points[-1]

            # Calculate the distance from the previous point
            # by using the AzimuthalEquidistant CRS centred on the previous point.
            distance_from_previous = ORIGIN.distance(
                previous.crs.project_geometry(point, src_crs=data_crs))

            points.append(TransectPoint(
                point=point,
                crs=self._crs_for_point(point, globe),
                distance_metres=previous.distance_metres + distance_from_previous,
                distance_normalised=self.line.project(point, normalized=True)
            ))

        return points

    @cached_property
    def segments(self) -> list[TransectSegment]:
        """
        A list of :class:`TransectSegments <.TransectSegment>`
        for each intersecting segment of the transect line and the dataset geometry.
        Segments are listed in order from the start of the line to the end of the line.
        """
        segments = []

        grid = self.convention.grids[self.convention.default_grid_kind]
        polygons = grid.geometry

        # Find all the cell polygons that intersect the line
        intersecting_indexes = grid.strtree.query(self.line, predicate='intersects')

        for linear_index in intersecting_indexes:
            polygon = polygons[linear_index]
            for intersection in self._intersect_polygon(polygon):
                # The line will have two ends.
                # The intersection starts and ends at these points.
                # Project those points alone the original line to find
                # the start and end distance of the intersection along the line.
                points = [
                    shapely.Point(intersection.coords[0]),
                    shapely.Point(intersection.coords[-1])
                ]
                projections: Iterable[tuple[shapely.Point, float]] = (
                    (point, self.distance_along_line(point))
                    for point in points)
                start, end = sorted(projections, key=lambda pair: pair[1])

                segments.append(TransectSegment(
                    start_point=start[0],
                    end_point=end[0],
                    intersection=intersection,
                    start_distance=start[1],
                    end_distance=end[1],
                    linear_index=linear_index,
                    polygon=polygon,
                ))

        return sorted(segments, key=lambda i: (i.start_distance, i.end_distance))

    @cached_property
    def coordinates(self) -> xarray.Dataset:
        """
        A :class:`xarray.Dataset` containing coordinate information
        for data extracted along the transect.
        """
        index_dim = 'index'
        coordinates = xarray.Dataset(
            coords={
                'distance': xarray.DataArray(
                    data=numpy.average(self.intersection_bounds, axis=1),
                    dims=index_dim,
                    attrs={
                        'long_name': 'Distance along transect',
                        'units': 'm',
                        'bounds': 'distance_bounds',
                    },
                ),
                'distance_bounds': xarray.DataArray(
                    data=self.intersection_bounds,
                    dims=(index_dim, 'Two'),
                ),
            }
        )
        return coordinates

    def _intersect_polygon(
        self,
        polygon: shapely.Polygon,
    ) -> list[shapely.LineString]:
        """
        Intersect a cell of the dataset geometry with the transect line,
        and return a list of all LineString segments of the intersection.
        This assumes that the cell does intersect the transect line.
        A line and a polygon can intersect in a number of ways:

        * a simple cut through the polygon
        * the line starts and/or stops in the polygon
        * the line intersects the polygon at a point
        * the line intersects the polygon multiple times

        Only the intersections that are line segments are returned.
        Multiple intersections (represented as a GeometryCollection)
        are decomposed in to the component geometries.
        Points are ignored.

        Parameters
        ----------
        polygon : shapely.Polygon
            The cell geometry to intersect

        Returns
        -------
        list of shapely.LineString
            All intersecting line strings
        """
        intersection = polygon.intersection(self.line)
        if isinstance(intersection, (shapely.GeometryCollection, shapely.MultiLineString)):
            geoms = intersection.geoms
        else:
            geoms = [intersection]
        return [geom for geom in geoms if isinstance(geom, shapely.LineString)]

    def distance_along_line(self, point: shapely.Point) -> float:
        """
        Calculate the distance in metres that the point
        falls along the :attr:`transect line <.line>`.
        If the point is not on the line,
        the point is projected on to the line
        and the distance is calculated to this point instead.

        This can be used to calculate the distance along the transect line
        to landmark features.
        These landmark features can be added as tick points along the transect.
        The landmark features need not fall directly on the line.

        Parameters
        ----------
        point : shapely.Point
            The point to calculate the distance to

        Returns
        -------
        float
            The distance the point is along the line in meters.
            If the point does not fall on the line,
            the point is first projected to the line.
        """
        data_crs = self.convention.data_crs
        distance_normalised = self.line.project(point, normalized=True)
        if distance_normalised < 0 or distance_normalised > 1:
            raise ValueError("Point is not on the line!")

        # Find the TransectPoint for the vertex before this point on the line
        line_point = next(
            lp for lp in reversed(self.points)
            if lp.distance_normalised <= distance_normalised)

        distance_from_point: float = ORIGIN.distance(
            line_point.crs.project_geometry(point, src_crs=data_crs))
        return line_point.distance_metres + distance_from_point

    def extract(self, data_array: DataArrayOrName) -> xarray.DataArray:
        """
        Extract data from a data array along a transect.

        Parameters
        ----------
        data_array : DataArrayOrName
            The data array to extract data from.

        Returns
        -------
        xarray.DataArray
            A new :class:`xarray.DataArray` containing data from the input data array
            extracted along the path of the transect.
        """
        data_array = name_to_data_array(self.dataset, data_array)

        # Some of the following operations drop attrs,
        # so keep a reference to the original ones
        attrs = data_array.attrs

        data_array = self.convention.ravel(data_array)

        index_dimension = data_array.dims[-1]
        data_array = move_dimensions_to_end(data_array, [index_dimension])

        data_array = data_array.isel({index_dimension: self.linear_indexes})

        # Restore attrs after reformatting
        data_array.attrs.update(attrs)

        return data_array

    def make_artist(
        self,
        axes: Axes,
        data_array: DataArrayOrName,
        **kwargs: Any,
    ) -> 'artists.TransectArtist':
        """
        Make an artist to plot values extracted from a data array along this transect.
        The kind of artist used depends on the dimensionality of the data array.

        To be plotted along a transect the data array must be defined on a supported :ref:`grid <grids>`.
        Currently only polygonal grids are supported.

        If a data array has a depth axis, :meth:`.make_cross_section_artist` is called,
        otherwise :meth:`.make_transect_step_artist` is called.

        Parameters
        ==========
        axes : Axes
            The :class:`matplotlib.axes.Axes` to add this artist to.
        data_array : DataArrayOrName
            The data array to plot
        **kwargs
            Passed on to the artist, can be used to customise the plot style.

        Returns
        =======
        :class:`.artists.TransectArtist`
            The artist that will plot the data.
            This artist will already have been added to the axes.

        See also
        ========
        :func:`~.utils.setup_distance_axis`
            Setup the x-axis of an :class:`~matplotlib.axes.Axes`
            for plotting distance along a transect.
        :func:`~.utils.setup_depth_axis`
            Setup the y-axis of an :class:`~matplotlib.axes.Axes`
            for plotting down a depth coordinate.
        """
        data_array = name_to_data_array(self.dataset, data_array)
        grid = self.convention.get_grid(data_array)
        try:
            depth_coordinate = self.convention.get_depth_coordinate_for_data_array(data_array)
        except NoSuchCoordinateError:
            depth_coordinate = None

        if grid.geometry_type is not shapely.Polygon:
            raise ValueError(
                f"I don't know how to plot transects across {grid.geometry_type.__name__} geometry.")

        if depth_coordinate is not None:
            return self.make_cross_section_artist(axes, data_array, **kwargs)
        else:
            return self.make_transect_step_artist(axes, data_array, **kwargs)

    def make_cross_section_artist(
        self,
        axes: Axes,
        data_array: DataArrayOrName,
        colorbar: bool = True,
        **kwargs: Any,
    ) -> 'artists.CrossSectionArtist':
        """
        Make an artist that plots a vertical slice along the length of the transect.
        The data must be three dimensional with a depth axis.
        The data are plotted as a grid of values,
        with distance along the transect as the x-axis
        and depth represented as the y-axis.

        Parameters
        ==========
        axes : Axes
            The :class:`matplotlib.axes.Axes` to add this line to.
        data_array : DataArrayOrName
            The data array to plot.
            This data array must be defined on a polygonal :class:`~emsarray.conventions.Grid`
            and must have a depth coordinate with bounds.
        colorbar : bool, default True
            Whether to add a colorbar for this artist.
            Sensible defaults are used for the colorbar, but if more customisation is required
            set `colorbar=False` and configure a colorbar manually.
        edgecolor : color, optional
            The colour of the line.
            Optional, defaults to the next available colour in the matplotlib plot colours.
        fill : bool, default False
            Whether to fill in values between the line and the baseline.
            Defaults to False.
        **kwargs
            Passed on to the :class:`~.artists.TransectStepArtist`,
            can be used to customise the plot.

        See also
        ========
        :func:`~.utils.setup_distance_axis`
            Setup the x-axis of an :class:`~matplotlib.axes.Axes`
            for plotting distance along a transect.
        :func:`~.utils.setup_depth_axis`
            Setup the y-axis of an :class:`~matplotlib.axes.Axes`
            for plotting down a depth coordinate.
        :func:`~emsarray.utils.estimate_bounds_1d`
            Estimate some bounds for a coordinate variable.
        """
        data_array = name_to_data_array(self.dataset, data_array)
        artist = artists.CrossSectionArtist.from_transect(
            self, data_array=data_array,
            **kwargs)
        axes.add_artist(artist)
        if colorbar:
            units = data_array.attrs.get('units', None)
            axes.figure.colorbar(artist, label=units)
        return artist

    def make_transect_step_artist(
        self,
        axes: Axes,
        data_array: DataArrayOrName,
        edgecolor: ColorType | None = 'auto',
        fill: bool = False,
        **kwargs: Any,
    ) -> 'artists.TransectStepArtist':
        """
        Make an artist that plots values along the length of the transect.
        The data must be two dimensional - it must have no depth axis.
        The data are plotted as a stepped line.

        Parameters
        ==========
        axes : Axes
            The :class:`matplotlib.axes.Axes` to add this line to.
        data_array : DataArrayOrName
            The data array to plot.
            This data array must be defined on a polygonal :class:`~emsarray.conventions.Grid`
            and must not have any other dimensions such as time or depth.
        edgecolor : color, optional
            The colour of the line.
            Optional, defaults to the next available colour in the matplotlib plot colours.
        fill : bool, default False
            Whether to fill in values between the line and the baseline.
            Defaults to False.
        **kwargs
            Passed on to the :class:`~.artists.TransectStepArtist`,
            can be used to customise the plot.

        Returns
        =======
        :class:`~.artists.TransectStepArtist`
            The artist that will plot the data.
            This artist will already have been added to the axes.

        See also
        ========
        :func:`.utils.setup_distance_axis`
            Setup the x-axis of an :class:`~matplotlib.axes.Axes`
            for plotting distance along a transect.
        """
        data_array = name_to_data_array(self.dataset, data_array)
        if edgecolor == 'auto':
            edgecolor = axes._get_lines.get_next_color()
        artist = artists.TransectStepArtist.from_transect(
            self, data_array=data_array,
            fill=fill, edgecolor=edgecolor, **kwargs)
        axes.add_artist(artist)
        return artist

    def make_ocean_floor_artist(
        self,
        axes: Axes,
        data_array: DataArrayOrName,
        fill: bool = True,
        facecolor: ColorType | None = 'lightgrey',
        edgecolor: ColorType | None = 'none',
        baseline: float | None = None,
        **kwargs: Any,
    ) -> 'artists.TransectStepArtist':
        """
        Make an artist that renders a solid polygon following a bathymetry variable.
        This can be drawn in front of a cross section artist to mask out values below the ocean floor.

        Parameters
        ==========
        axes : Axes
            The :class:`matplotlib.axes.Axes` to add the ocean floor artist to
        data_array : DataArrayOrName
            The data array or name of a data array with the ocean floor data
        baseline : float, optional
            The deepest part of the ocean floor to render.
            The ocean floor will be filled in from the bathymetry value down to the baseline.
            Optional, if not provided the deepest value in the data array is used instead.
        **kwargs
            Passed on to the :class:`.artists.TransectStepArtist` for styling.
            Set `facecolor` to change the colour of the ocean floor polygon.

        Returns
        =======
        .artists.TransectStepArtist
            The artist that will render the ocean floor.
            This artist will already have been added to the axes.

        Notes
        =====
        The `sign convention <CF-vertical-coordinates>`_
        of the bathymetry variable and the depth coordinate must match.
        If they differ the ocean floor polygon is likely to be either
        entirely outside of the plot extent or to cover the entire plot extent.
        :func:`~emsarray.operations.depth.normalize_depth_variables`
        can be used to change the sign convention of a depth coordinate variable.

        See also
        ========
        `CF Conventions on Vertical Coordinates <CF-vertical-coordinates>`_
            More information on the `positive` attribute.
        :func:`emsarray.operations.depth.normalize_depth_variables`
            Update the sign convention of a depth coordinate variable.

        .. _CF-vertical-coordinates: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#vertical-coordinate

        """
        data_array = name_to_data_array(self.dataset, data_array)
        if baseline is None:
            data_min, data_max = numpy.nanmin(data_array.values), numpy.nanmax(data_array.values)
            if 'positive' in data_array.attrs:
                if data_array.attrs['positive'] == 'down':
                    baseline = data_max
                else:
                    baseline = data_min
            else:
                # Take a guess by using the most extreme value
                if numpy.abs(data_min) < numpy.abs(data_max):
                    baseline = data_max
                else:
                    baseline = data_min

        artist = artists.TransectStepArtist.from_transect(
            self, data_array=data_array,
            fill=fill, baseline=baseline,
            facecolor=facecolor, edgecolor=edgecolor,
            **kwargs)
        axes.add_artist(artist)
        return artist
