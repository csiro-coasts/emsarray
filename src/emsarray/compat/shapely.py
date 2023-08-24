import warnings
from typing import Generic, Iterable, Tuple, TypeVar, Union, cast

import numpy
import shapely
from packaging.version import parse
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

shapely_version = parse(shapely.__version__)
v2 = parse("2")
v1_8_3 = parse("1.8.3")

T = TypeVar("T")


class SpatialIndex(Generic[T]):
    """
    A wrapper around a shapely STRtree that can associate metadata with
    geometries.

    This also handles the version differences in STRtree between
    shapely ~= 1.8.x and shapely >= 2.0.0
    """
    items: numpy.ndarray
    index: STRtree

    dtype: numpy.dtype = numpy.dtype([('geom', numpy.object_), ('data', numpy.object_)])

    def __init__(
        self,
        items: Union[numpy.ndarray, Iterable[Tuple[BaseGeometry, T]]],
    ):
        self.items = numpy.array(items, dtype=self.dtype)

        if shapely_version >= v2:
            self.index = STRtree(self.items['geom'])
        elif shapely_version >= v1_8_3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ShapelyDeprecationWarning)
                self.index = STRtree(geoms=self.items['geom'])
        else:
            self.index = STRtree(geoms=self.items['geom'])

    def query(
        self,
        geom: BaseGeometry,
    ) -> numpy.ndarray:
        if shapely_version >= v2:
            indices = self.index.query(geom)
        else:
            indices = self.index._query(geom)
        return cast(numpy.ndarray, self.items.take(indices))

    def nearest(
        self,
        geom: BaseGeometry,
    ) -> numpy.ndarray:
        if shapely_version >= v2:
            indices = self.index.nearest(geom)
        else:
            indices = self.index._nearest(geom)
        return cast(numpy.ndarray, self.items.take(indices))
