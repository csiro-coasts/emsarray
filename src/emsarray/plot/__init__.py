from typing import Any

from emsarray.utils import RequiresExtraException, requires_extra

try:
    from . import utils
    from .base import animate_on_figure, plot_on_figure
    from .shortcuts import add_coast, add_gridlines, add_landmarks
    from .utils import (
        bounds_to_extent, make_plot_title, polygons_to_collection
    )

    CAN_PLOT = True
    IMPORT_EXCEPTION = None

except ImportError as exc:
    CAN_PLOT = False
    IMPORT_EXCEPTION = exc

    def __getattr__(name: str) -> Any:
        # module.__getattr__("name") is called when code accessed module.name,
        # but the module doesn't have that attribute set.
        # We use this to raise a RequiresExtraException with the assumption
        # that importing something failed.

        if name in __all__:
            raise RequiresExtraException("plot") from IMPORT_EXCEPTION
        else:
            raise AttributeError(name)


# The external API of this module.
__all__ = [
    'CAN_PLOT', '_requires_plot',

    # Export the utils module explicitly. Future utility methods may not be
    # exported using this public API, and instead be accessed using `plot.utils.some_method`
    'utils',

    # Methods from .base
    'animate_on_figure', 'plot_on_figure',

    # Methods from .utils
    'add_coast', 'add_gridlines', 'add_landmarks', 'polygons_to_collection',
    'bounds_to_extent', 'make_plot_title'
]


_requires_plot = requires_extra(extra='plot', import_error=IMPORT_EXCEPTION)
