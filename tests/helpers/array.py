import itertools

import numpy


def reduce_axes(arr: numpy.ndarray, axes: tuple[bool, ...] | None = None) -> numpy.ndarray:
    """
    Reduce the size of an array by one on an axis-by-axis basis. If an axis is
    reduced, neigbouring values are averaged together

    :param arr: The array to reduce.
    :param axes: A tuple of booleans indicating which axes should be reduced. Optional, defaults to reducing along all axes.
    :returns: A new array with the same number of axes, but one size smaller in each axis that was reduced.
    """
    if axes is None:
        axes = tuple(True for _ in arr.shape)
    axes_slices = [[numpy.s_[+1:], numpy.s_[:-1]] if axis else [numpy.s_[:]] for axis in axes]
    return numpy.mean([arr[tuple(p)] for p in itertools.product(*axes_slices)], axis=0)  # type: ignore


def mask_from_strings(mask_strings: list[str]) -> numpy.ndarray:
    """
    Make a boolean mask array from a list of strings:

        >>> mask_from_strings([
        ...     "101",
        ...     "010",
        ...     "111",
        ... ])
        array([[ True, False,  True],
               [False,  True, False],
               [ True,  True,  True]])
    """
    return numpy.array([list(map(int, line)) for line in mask_strings]).astype(bool)
