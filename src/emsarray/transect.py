from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import (
    Any, Callable, Generic, Hashable, Iterable, List, Optional, Tuple, Union
)

import cfunits
import numpy
import shapely
import xarray
from cartopy import crs
from matplotlib import animation, cm, pyplot
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, Formatter

from emsarray.conventions import Convention, Index
from emsarray.plot import _requires_plot
from emsarray.types import Landmark
from emsarray.utils import move_dimensions_to_end

# Useful for calculating distances in a AzimuthalEquidistant projection
# centred on some point:
#
#     az = crs.AzimuthalEquidistant(p1.x, p1.y)
#     distance = az.project_geometry(p2).distance(ORIGIN)
ORIGIN = shapely.Point(0, 0)


def plot(
    dataset: xarray.Dataset,
    line: shapely.LineString,
    data_array: xarray.DataArray,
    *,
    figsize: tuple = (12, 3),
    **kwargs: Any,
) -> Figure:
    """
    Plot a transect of a dataset.

    This is convenience function that handles the most common use cases.
    For more options refer to the :class:`.Transect` class.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to transect.
    line : shapely.LineString
        The transect path to plot.
    data_array : xarray.DataArray
        A variable from the dataset to plot.
    figsize : tuple of int, int
        The size of the figure.
    **kwargs
        Passed to :meth:`Transect.plot_on_figure()`.
    """
    figure = pyplot.figure(layout="constrained", figsize=figsize)
    transect = Transect(dataset, line)
    transect.plot_on_figure(figure, data_array, **kwargs)
    figure.show()
    return figure


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
class TransectSegment(Generic[Index]):
    """
    A TransectSegment holds information about each intersecting segment of the
    transect path and the dataset cells.
    """
    start_point: shapely.Point
    end_point: shapely.Point
    intersection: shapely.LineString
    start_distance: float
    end_distance: float
    index: Index
    linear_index: int
    polygon: shapely.Polygon


class Transect:
    """
    """
    #: The dataset to plot a transect through
    dataset: xarray.Dataset

    #: The transect path to plot
    line: shapely.LineString

    #: The depth coordinate (or the name of the depth coordinate) for the dataset.
    depth: xarray.DataArray

    def __init__(
        self,
        dataset: xarray.Dataset,
        line: shapely.LineString,
        depth: Optional[Union[Hashable, xarray.DataArray]] = None,
    ):
        self.dataset = dataset
        self.convention = dataset.ems
        self.line = line
        if depth is not None:
            self.depth = self.convention._get_data_array(depth)
        else:
            self.depth = self.convention.depth_coordinate

    @cached_property
    def convention(self) -> Convention:
        convention: Convention = self.dataset.ems
        return convention

    @cached_property
    def transect_dataset(self) -> xarray.Dataset:
        """
        A :class:`~xarray.Dataset` containing all the transect geometry.
        This includes the depth data, path lengths,
        and the linear index of each intersecting cell in the source dataset.
        This transect dataset contains all the information necessary to generate a plot,
        except for the actual variable data being plotted.
        """
        depth = self.depth

        depth_dimension = depth.dims[0]

        depth_bounds = None
        try:
            depth_bounds = self.convention.dataset[depth.attrs['bounds']].values
        except KeyError:
            # Make up some depth bounds data from the depth values
            # The top/bottom values will be the first/last depth values,
            # all other points are the midpoints between the neighbouring points.
            depth_midpoints = numpy.concatenate([
                [depth.values[0]],
                (depth.values[1:] + depth.values[:-1]) / 2,
                [depth.values[-1]]
            ])
            depth_bounds = numpy.column_stack((
                depth_midpoints[:-1],
                depth_midpoints[1:],
            ))

        try:
            positive_down = depth.attrs['positive'] == 'down'
        except KeyError as err:
            raise ValueError(
                f'Depth variable {depth.name!r} must have a `positive` attribute'
            ) from err

        linear_indices = [segment.linear_index for segment in self.segments]
        depth = xarray.DataArray(
            data=depth.values,
            dims=(depth_dimension,),
            attrs={
                'bounds': 'depth_bounds',
                'positive': 'down' if positive_down else 'up',
                'long_name': depth.attrs.get('long_name'),
                'description': depth.attrs.get('description'),
                'units': depth.attrs.get('units'),
            },
        )
        depth_bounds = xarray.DataArray(
            data=depth_bounds,
            dims=(depth_dimension, 'bounds'),
        )
        distance_bounds = xarray.DataArray(
            data=[
                [segment.start_distance, segment.end_distance]
                for segment in self.segments
            ],
            dims=('index', 'bounds'),
            attrs={
                'long_name': 'Distance along transect',
                'units': 'm',
                'start_distance': self.points[0].distance_metres,
                'end_distance': self.points[-1].distance_metres,
            },
        )
        linear_index = xarray.DataArray(
            data=linear_indices,
            dims=('index',)
        )

        return xarray.Dataset(
            data_vars={
                'depth_bounds': depth_bounds,
                'distance_bounds': distance_bounds,
            },
            coords={
                'depth': depth,
                'linear_index': linear_index,
            },
        )

    def _set_up_axis(self, variable: xarray.DataArray) -> Tuple[str, Formatter]:
        title = str(variable.attrs.get('long_name'))
        units: Optional[str] = variable.attrs.get('units')

        if units is not None:
            # Use cfunits to normalize the units to their short symbol form.
            # EngFormatter will write 'k{unit}', 'G{unit}', etc
            # so unit symbols are required.
            units = cfunits.Units(units).formatted()
            formatter = EngFormatter(unit=units)

        return title, formatter

    def _crs_for_point(
        self,
        point: shapely.Point,
        globe: Optional[crs.Globe] = None,
    ) -> crs.Projection:
        return crs.AzimuthalEquidistant(
            central_longitude=point.x, central_latitude=point.y, globe=globe)

    @cached_property
    def points(
        self,
    ) -> List[TransectPoint]:
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
    def segments(self) -> List[TransectSegment[Index]]:
        """
        A list of :class:`.TransectSegmens` for each intersecting segment of the transect line and the dataset geometry.
        Segments are listed in order from the start of the line to the end of the line.
        """
        segments = []

        # Find all the cell polygons that intersect the line
        intersecting_indices = self.convention.strtree.query(self.line, predicate='intersects')

        for linear_index in intersecting_indices:
            polygon = self.convention.polygons[linear_index]
            index = self.convention.wind_index(linear_index)
            for intersection in self._intersect_polygon(polygon):
                # The line will have two ends.
                # The intersection starts and ends at these points.
                # Project those points alone the original line to find
                # the start and end distance of the intersection along the line.
                points = [
                    shapely.Point(intersection.coords[0]),
                    shapely.Point(intersection.coords[-1])
                ]
                projections: Iterable[Tuple[shapely.Point, float]] = (
                    (point, self.distance_along_line(point))
                    for point in points)
                start, end = sorted(projections, key=lambda pair: pair[1])

                segments.append(TransectSegment(
                    start_point=start[0],
                    end_point=end[0],
                    intersection=intersection,
                    start_distance=start[1],
                    end_distance=end[1],
                    index=index,
                    linear_index=linear_index,
                    polygon=polygon,
                ))

        return sorted(segments, key=lambda i: (i.start_distance, i.end_distance))

    def _intersect_polygon(
        self,
        polygon: shapely.Polygon,
    ) -> List[shapely.LineString]:
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

    def make_poly_collection(
        self,
        **kwargs: Any,
    ) -> PolyCollection:
        """
        Make a :class:`matplotlib.collections.PolyCollection`
        representing the transect geometry.

        Parameters
        ----------
        **kwargs
            Any keyword arguments are passed to the PolyCollection constructor.

        Returns
        -------
        matplotlib.collections.PolyCollection
            A PolyCollection representing all the cells
            and all the depths the transect line intesected.
        """
        transect_dataset = self.transect_dataset
        distance_bounds = transect_dataset['distance_bounds'].values
        depth_bounds = transect_dataset['depth_bounds'].values
        vertices = [
            [
                (distance_bounds[index, 0], depth_bounds[depth_index][0]),
                (distance_bounds[index, 0], depth_bounds[depth_index][1]),
                (distance_bounds[index, 1], depth_bounds[depth_index][1]),
                (distance_bounds[index, 1], depth_bounds[depth_index][0]),
            ]
            for depth_index in range(transect_dataset.coords['depth'].size)
            for index in range(transect_dataset.dims['index'])
        ]
        return PolyCollection(vertices, **kwargs)

    def make_ocean_floor_poly_collection(
        self,
        bathymetry: xarray.DataArray,
        **kwargs: Any
    ) -> PolyCollection:
        """
        Make a :class:`matplotlib.collections.PolyCollection`
        representing the ocean floor.
        This can be overlayed on a transect plot to mask out values below the sea floor.

        Parameters
        ----------
        bathymetry : xarray.Dataset
            A data array containing bathymetry data for the dataset.
        **kwargs
            Any keyword arguments are passed on to the
            :class:`~matplotlib.collections.PolyCollection` constructor

        Returns
        -------
        matplotlib.collections.PolyCollection
            A collection of polygons representing
            the ocean floor along the transect path.
        """
        transect_dataset = self.transect_dataset
        depth = transect_dataset['depth']

        bathymetry_values = self.convention.ravel(bathymetry)
        # The bathymetry data can be oriented differently to the depth coordinate.
        # Correct for this if so.
        if 'positive' in bathymetry.attrs:
            if bathymetry.attrs['positive'] != depth.attrs['positive']:
                bathymetry_values = -bathymetry_values

        positive_down = depth.attrs['positive'] == 'down'
        deepest_fn = numpy.nanmax if positive_down else numpy.nanmin
        deepest = deepest_fn(bathymetry_values.values)

        distance_bounds = transect_dataset['distance_bounds'].values
        linear_indices = transect_dataset['linear_index'].values

        vertices = [
            [
                (distance_bounds[index, 0], bathymetry_values[linear_indices[index]]),
                (distance_bounds[index, 0], deepest),
                (distance_bounds[index, 1], deepest),
                (distance_bounds[index, 1], bathymetry_values[linear_indices[index]]),
            ]
            for index in range(transect_dataset.dims['index'])
        ]
        return PolyCollection(vertices, **kwargs)

    def prepare_data_array_for_transect(self, data_array: xarray.DataArray) -> xarray.DataArray:
        """
        Prepare a data array for being used as the data in a transect plot.

        Parameters
        ----------
        data_array : xarray.DataArray
            The data array that will be plotted

        Returns
        -------
        xarray.DataArray
            The input data array transformed to have the correct shape
            for plotting on the transect.
        """
        data_array = self.convention.ravel(data_array)

        depth_dimension = self.transect_dataset.coords['depth'].dims[0]
        index_dimension = data_array.dims[-1]
        data_array = move_dimensions_to_end(data_array, [depth_dimension, index_dimension])

        linear_indices = self.transect_dataset['linear_index'].values
        data_array = data_array.isel({index_dimension: linear_indices})

        return data_array

    def _find_depth_bounds(self, data_array: xarray.DataArray) -> Tuple[int, int]:
        """
        Find the shallowest and deepest layers of the data array
        where there is at least one value per depth.

        Most ocean models represent cells that are below the sea floor as nans.
        Some ocean models do the same for layers above the sea surface,
        which can vary due to tides.
        If a transect covers mostly shallow regions
        but the dataset includes very deep layers
        the shallow regions become very small on the final plot.

        This function finds the indices of the deepest and shallowest layers
        where the values are not entirely nan
        along the transect path.
        The transect plot can use these to only plot depth values that have data,
        trimming off layers that are nothing but ocean floor.
        """
        transect_dataset = self.transect_dataset
        dim = transect_dataset['depth'].dims[0]

        start = 0
        for index in range(transect_dataset['depth'].size):
            if numpy.any(numpy.isfinite(data_array.isel({dim: index}).values)):
                start = index
                break

        stop = -1
        for index in reversed(range(transect_dataset['depth'].size)):
            if numpy.any(numpy.isfinite(data_array.isel({dim: index}))):
                stop = index
                break

        return start, stop

    @_requires_plot
    def plot_on_figure(
        self,
        figure: Figure,
        data_array: xarray.DataArray,
        *,
        title: Optional[str] = None,
        trim_nans: bool = True,
        clamp_to_surface: bool = True,
        bathymetry: Optional[xarray.DataArray] = None,
        cmap: Union[str, Colormap] = 'jet',
        ocean_floor_colour: str = 'black',
        landmarks: Optional[List[Landmark]] = None,
    ) -> None:
        """
        Plot the data array along this transect.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The figure to plot on
        data_array : xarray.DataArray
            The data array to plot.
            This should be a data array from the dataset provided to the
            Transect constructor,
            or a data array of compatible shape.
        title : str, optional
            The title of the plot.
            Defaults to the 'long_name' attribute of the data array.
        trim_nans : bool, default True
            Whether to trim layers containing all nans.
            Layers that are entirely under the ocean floor are often represented as nans.
            Without trimming, transects through shallow areas mostly look like ocean floor.
        clamp_to_surface : bool, default True
            If true, clamp the y-axis to 0 m.
            Some datasets define an upper depth bound of some large number
            which rather spoils the plot.
        bathymetry : xarray.DataArray, optional
            A data array containing bathymetry information for the dataset.
            This will be used to draw a more detailed ocean floor mask.
        ocean_floor_colour : str, default 'grey'
            The colour to draw the ocean floor in.
            This is used to draw cells containing nan values,
            and the bathymetry data.
        landmarks : list of str, :class:`shapely.Point` tuples
            A list of (name, point) tuples.
            These will be added as tick marks along the top of the plot.
        """
        axes, collection, data_array = self._plot_on_figure(
            figure=figure,
            data_array=data_array,
            title=title,
            trim_nans=trim_nans,
            clamp_to_surface=clamp_to_surface,
            bathymetry=bathymetry,
            cmap=cmap,
            ocean_floor_colour=ocean_floor_colour,
            landmarks=landmarks,
        )
        collection.set_array(data_array.values.flatten())

    def animate_on_figure(
        self,
        figure: Figure,
        data_array: xarray.DataArray,
        *,
        title: Optional[Union[str, Callable[[Any], str]]] = None,
        trim_nans: bool = True,
        clamp_to_surface: bool = True,
        bathymetry: Optional[xarray.DataArray] = None,
        cmap: Union[str, Colormap] = 'jet',
        ocean_floor_colour: str = 'black',
        landmarks: Optional[List[Landmark]] = None,
        coordinate: Optional[xarray.DataArray] = None,
        interval: int = 200,
    ) -> animation.FuncAnimation:
        """
        Plot the data array along this transect.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The figure to plot on
        data_array : xarray.DataArray
            The data array to plot.
            This should be a data array from the dataset provided to the
            Transect constructor,
            or a data array of compatible shape.
        title : str or callable
            The title of the plot.
        coordinate : xarray.DataArray
            The coordinate to animate along.
            Defaults to the time coordinate.
        interval : int
            Time in milliseconds between frames.
        **kwargs
            See :meth:`.plot_on_figure` for available keyword arguments
        """
        if coordinate is None:
            coordinate = self.convention.time_coordinate
        coordinate_indexes = numpy.arange(coordinate.size)

        coordinate_callable: Callable[[Any], str]
        if title is None:
            title = data_array.attrs.get('long_name')
            if title is not None:
                coordinate_callable = lambda c: f'{title}\n{c}'
            else:
                coordinate_callable = str

        elif isinstance(title, str):
            coordinate_callable = title.format

        else:
            coordinate_callable = title

        axes, collection, data_array = self._plot_on_figure(
            figure=figure,
            data_array=data_array,
            title=None,
            trim_nans=trim_nans,
            clamp_to_surface=clamp_to_surface,
            bathymetry=bathymetry,
            cmap=cmap,
            ocean_floor_colour=ocean_floor_colour,
            landmarks=landmarks,
        )

        def animate(index: int) -> Iterable[Artist]:
            changes: List[Artist] = []

            coordinate_value = coordinate.values[index]
            axes.set_title(coordinate_callable(coordinate_value))
            changes.append(axes)

            collection.set_array(data_array[index].values.flatten())
            changes.append(collection)
            return changes

        # Draw the figure to force everything to compute its size
        figure.draw_without_rendering()

        # Set the first frame of data
        animate(0)

        # Make the animation
        return animation.FuncAnimation(
            figure, animate, frames=coordinate_indexes,
            interval=interval)

    def _plot_on_figure(
        self,
        figure: Figure,
        data_array: xarray.DataArray,
        *,
        title: Optional[str] = None,
        trim_nans: bool = True,
        clamp_to_surface: bool = True,
        bathymetry: Optional[xarray.DataArray] = None,
        cmap: Union[str, Colormap] = 'jet',
        ocean_floor_colour: str = 'black',
        landmarks: Optional[List[Landmark]] = None,
    ) -> Tuple[Axes, PolyCollection, xarray.DataArray]:
        """
        Construct the axes and PolyCollections on a plot,
        and reformat the data array to the correct shape for plotting.
        Assigning the data is left to the caller,
        to support both static and animated plots.
        """
        transect_dataset = self.transect_dataset
        depth = transect_dataset.coords['depth']
        distance_bounds = transect_dataset.data_vars['distance_bounds']

        data_array = self.prepare_data_array_for_transect(data_array)

        positive_down = depth.attrs['positive'] == 'down'
        d1, d2 = depth.values[0:2]
        deep_to_shallow = (d1 > d2) == positive_down

        if trim_nans:
            depth_start, depth_stop = self._find_depth_bounds(data_array)
        else:
            depth_start, depth_stop = 0, -1
        if deep_to_shallow:
            depth_start, depth_stop = depth_stop, depth_start

        down, up = (
            (numpy.nanmax, numpy.nanmin)
            if positive_down
            else (numpy.nanmin, numpy.nanmax))
        if clamp_to_surface:
            depth_limit_shallow = 0
        else:
            depth_limit_shallow = up(transect_dataset['depth_bounds'][depth_start])
        depth_limit_deep = down(transect_dataset['depth_bounds'][depth_stop])

        axes = figure.subplots()
        x_title, x_formatter = self._set_up_axis(distance_bounds)
        y_title, y_formatter = self._set_up_axis(depth)
        axes.set_xlabel(x_title)
        axes.set_ylabel(y_title)
        axes.xaxis.set_major_formatter(x_formatter)
        axes.yaxis.set_major_formatter(y_formatter)
        axes.set_xlim(
            distance_bounds.attrs['start_distance'],
            distance_bounds.attrs['end_distance'],
        )
        axes.set_ylim(depth_limit_deep, depth_limit_shallow)

        if title is None:
            title = data_array.attrs.get('long_name')
        if title is not None:
            axes.set_title(title)

        cmap = cm.get_cmap(cmap).copy()
        cmap.set_bad(ocean_floor_colour)
        collection = self.make_poly_collection(
            cmap=cmap, clim=(numpy.nanmin(data_array), numpy.nanmax(data_array)))
        axes.add_collection(collection)

        if bathymetry is not None:
            ocean_floor = self.make_ocean_floor_poly_collection(
                bathymetry, facecolor=ocean_floor_colour)
            axes.add_collection(ocean_floor)

        units = data_array.attrs.get('units')
        figure.colorbar(collection, ax=axes, location='right', label=units)

        if landmarks is not None:
            top_axis = axes.secondary_xaxis('top')
            top_axis.set_ticks(
                [self.distance_along_line(point) for label, point in landmarks],
                [label for label, point in landmarks],
            )

        return axes, collection, data_array
