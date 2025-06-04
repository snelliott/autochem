"""Plotting helpers."""

import itertools
from collections.abc import Sequence
from typing import Protocol

import altair
import numpy
import pandas
from numpy.typing import ArrayLike, NDArray

from .. import unit_
from ..unit_ import UNITS, Units, UnitsData


class RateFunction(Protocol):
    """Protocol for callable rate function."""

    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float128]: ...


class Color:
    black = "#000000"
    blue = "#0066ff"
    red = "#ff0000"
    green = "#1ab73a"
    orange = "#ef7810"
    purple = "#8533ff"
    pink = "#d0009a"
    yellow = "#ffcd00"
    brown = "#916e6e"


COLOR_CYCLE = [
    Color.blue,
    Color.red,
    Color.green,
    Color.orange,
    Color.purple,
    Color.pink,
    Color.yellow,
    Color.brown,
]


def arrhenius(
    ks: Sequence[RateFunction] = (),
    labels: Sequence[str] = (),
    T_range: tuple[float, float] = (400, 1250),  # noqa: N803
    P: float = 1,  # noqa: N803
    order: int = 1,
    units: UnitsData | None = None,
    x_label: str = "1000/𝑇",  # noqa: RUF001
    y_label: str = "𝑘",
) -> altair.Chart:
    """Display as an Arrhenius plot.

    :param others: Other rate constants
    :param others_labels: Labels for other rate constants
    :param T_range: Temperature range
    :param P: Pressure
    :param units: Units
    :param x_label: X-axis label
    :param y_label: Y-axis label
    :return: Chart
    """
    assert len(ks) == len(labels), f"{labels} !~ {ks}"
    nk = len(ks)
    colors = list(itertools.islice(itertools.cycle(COLOR_CYCLE), nk))

    # Process units
    units = UNITS if units is None else Units.model_validate(units)
    x_unit = unit_.pretty_string(units.temperature**-1)
    y_unit = unit_.pretty_string(units.rate_constant(order))

    # Add units to labels
    x_label = f"{x_label} ({x_unit})"
    y_label = f"{y_label} ({y_unit})"

    # Gather data from functons
    T = numpy.linspace(*T_range, num=500)
    data_dct = {lb: k(T, P, units=units) for lb, k in zip(labels, ks, strict=True)}
    data = pandas.DataFrame({"x": numpy.divide(1000, T), **data_dct})

    # Determine exponent range
    vals_arr = numpy.array(list(data_dct.values()))
    is_nan = numpy.isnan(vals_arr)
    exp_arr = numpy.log10(vals_arr, where=~is_nan)
    exp_arr[is_nan] = 0.0
    exp_arr = numpy.rint(exp_arr).astype(int)
    exp_max = numpy.max(exp_arr).item()
    exp_min = numpy.min(exp_arr).item()
    y_vals = [10**x for x in range(exp_min, exp_max + 2)]

    # Prepare encoding parameters
    x = altair.X("x", title=x_label)
    y = (
        altair.Y("value:Q", title=y_label)
        .scale(type="log")
        .axis(format=".1e", values=y_vals)
    )
    color = (
        altair.Color("key:N", scale=altair.Scale(domain=labels, range=colors))
        if nk > 1
        else altair.value(colors[0])
    )

    # Create chart
    return (
        altair.Chart(data)
        .mark_line()
        .transform_fold(fold=list(data_dct.keys()))
        .encode(x=x, y=y, color=color)
    )
