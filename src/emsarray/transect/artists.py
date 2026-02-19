from typing import Any

import numpy
import xarray
from matplotlib.artist import Artist
from matplotlib.collections import QuadMesh
from matplotlib.patches import StepPatch

from . import base


class TransectArtist(Artist):
    """
    A matplotlib Artist subclass that knows what Transect it is associated with,
    and has a `set_data_array()` method.
    Users can call `TransectArtist.set_data_array()` to update the data in a plot.
    This is useful when making animations, for example.
    """
    _transect: 'base.Transect'

    def set_transect(self, transect: 'base.Transect') -> None:
        if hasattr(self, '_transect'):
            raise ValueError("_transect can not be changed once set")
        self._transect = transect

    def get_transect(self) -> 'base.Transect':
        return self._transect

    def set_data_array(self, data_array: Any) -> None:
        """
        Update the data this artist is plotting.
        """
        raise NotImplementedError("Subclasses must implement this")


class CrossSectionArtist(QuadMesh, TransectArtist):
    @classmethod
    def from_transect(
        cls,
        transect: "base.Transect",
        *,
        data_array: xarray.DataArray | None = None,
        depth_coordinate: xarray.DataArray | None = None,
        **kwargs: Any,
    ) -> "CrossSectionArtist":
        """
        Construct a :class:`CrossSectionArtist` for a transect.
        """
        distance_bounds = transect.intersection_bounds

        if depth_coordinate is None and data_array is None:
            raise ValueError(
                "At least one of data_array and depth_coordinate must be not None")
        if depth_coordinate is None:
            depth_coordinate = transect.convention.get_depth_coordinate_for_data_array(data_array)
        depth_bounds = transect.dataset[depth_coordinate.attrs['bounds']].values

        holes = transect.holes
        xs = numpy.concat([distance_bounds[:, 0], distance_bounds[-1:, 1]])
        xs = numpy.insert(xs, holes, distance_bounds[holes - 1, 1])
        ys = numpy.concat([depth_bounds[:, 0], depth_bounds[-1:, 1]])
        coordinates = numpy.stack(numpy.meshgrid(xs, ys), axis=-1)

        # There are issues with passing both transect and data array to the constructor
        # where the `set_data_array()` is called before `set_transect()`.
        # Doing it this way is safe but kinda gross.
        artist = cls(coordinates, transect=transect, **kwargs)
        if data_array is not None:
            artist.set_data_array(data_array)

        return artist

    def set_data_array(self, data_array: xarray.DataArray) -> None:
        if len(self._transect.segments) > 0:
            self.set_array(self.prepare_data_array(self._transect, data_array))

    @staticmethod
    def prepare_data_array(transect: "base.Transect", data_array: xarray.DataArray) -> numpy.ndarray:
        values = transect.extract(data_array).values
        values = numpy.insert(values, transect.holes, numpy.nan, axis=-1)
        return values


class TransectStepArtist(StepPatch, TransectArtist):
    _edge_default = True

    @classmethod
    def from_transect(
        cls,
        transect: "base.Transect",
        *,
        data_array: xarray.DataArray | None = None,
        **kwargs: Any,
    ) -> "TransectStepArtist":
        """
        Construct a :class:`TransectStepArtist` for a transect.
        """
        holes = transect.holes
        x_bounds = transect.intersection_bounds
        if len(transect.segments) == 0:
            edges = numpy.array([0.])
        else:
            edges = x_bounds[:, 0]
            edges = numpy.append(edges, x_bounds[-1, 1])
            edges = numpy.insert(edges, holes, x_bounds[holes - 1, 1])

        if data_array is not None:
            values = cls.prepare_data_array(transect, data_array)
        else:
            values = numpy.full(shape=(len(edges) - 1,), fill_value=numpy.nan)

        return cls(values, edges, transect=transect, **kwargs)

    def set_data_array(self, data_array: xarray.DataArray) -> None:
        self.set_data(self.prepare_data_array(self._transect, data_array))

    @staticmethod
    def prepare_data_array(transect: "base.Transect", data_array: xarray.DataArray) -> numpy.ndarray:
        values = transect.extract(data_array).values
        assert len(values.shape) == 1

        # If a transect path is not fully contained within the dataset geometry
        # the path will have gaps. We can represent these gaps using nans.
        values = numpy.insert(
            values.astype(float),  # Upcast to float in case this was an integer array
            transect.holes,
            numpy.nan,
        )

        return values
