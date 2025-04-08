"""Rate constant models."""

import abc
from collections.abc import Mapping, Sequence
from typing import Annotated, ClassVar

import altair
import more_itertools as mit
import numpy
import pandas
import pint
import pydantic
import xarray
from numpy.polynomial import chebyshev
from numpy.typing import ArrayLike, NDArray
from pydantic_core import core_schema

from .. import unit_
from ..unit_ import UNITS, Dimension, UnitManager, Units, UnitsData
from ..util import chemkin
from ..util.type_ import Frozen, NDArray_, Scalable, Scalers, SubclassTyped
from ._00func import (
    BlendingFunction_,
    extract_blending_function_from_chemkin_parse_results,
)
from ._00func import (
    chemkin_aux_lines as blending_function_chemkin_aux_lines,
)

COLOR_SEQUENCE = [
    "#0066ff",  # blue
    "#ff0000",  # red
    "#1ab73a",  # green
    "#ef7810",  # orange
    "#8533ff",  # purple
    "#d0009a",  # pink
    "#ffcd00",  # yellow
    "#916e6e",  # brown
]


class BaseRateConstant(UnitManager, Frozen, Scalable, SubclassTyped, abc.ABC):
    """Abstract base class for rate constants."""

    order: int = 1

    @property
    def unit(self) -> pint.Unit:
        return UNITS.rate_constant(self.order)

    def __init__(self, units: UnitsData | None = None, **kwargs):
        super().__init__(units=units, **kwargs)

    @abc.abstractmethod
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate rate constant.

        :param T: Temperature(s)
        :param P: Pressure(s)
        :param units: Input units and desired output units
        :return: Value(s)
        """
        pass

    def process_input(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike,  # noqa: N803
    ) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """Normalize rate constant input.

        :param T: Temperature(s)
        :param P: Pressure(s)
        :return: Temperature(s) and pressure(s)
        """
        T, P = numpy.meshgrid(T, P)
        return T, P

    def process_output(
        self,
        kTP: ArrayLike,  # noqa: N803
        T: ArrayLike,  # noqa: N803
        P: ArrayLike,  # noqa: N803
    ) -> NDArray[numpy.float64]:
        """Normalize rate constant output, clipping unphyiscal negative values.

        :param ktp: Rate constant values
        :return: Rate constant values
        """
        kTP = numpy.reshape(kTP, numpy.shape(T) + numpy.shape(P))
        return numpy.where(numpy.less_equal(kTP, 0), numpy.nan, kTP)

    def display(
        self,
        others: "Sequence[BaseRateConstant]" = (),
        labels: Sequence[str] = (),
        T_range: tuple[float, float] = (400.0, 1250.0),  # noqa: N803
        P: float = 1,  # noqa: N803
        units: UnitsData | None = None,
        label: str = "This work",
        x_label: str = "1000/T",
        y_label: str = "k",
    ) -> altair.Chart:
        """Display as an Arrhenius plot.

        :param others: Other rate constants
        :param others_labels: Labels for other rate constants
        :param t_range: Temperature range
        :param P: Pressure
        :param units: Units
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :return: Chart
        """
        # Process units
        units = UNITS if units is None else Units.model_validate(units)
        x_unit = unit_.pretty_string(units.temperature**-1)
        y_unit = unit_.pretty_string(units.rate_constant(self.order))

        # Add units to labels
        x_label = f"{x_label} ({x_unit})"
        y_label = f"{y_label} ({y_unit})"

        # Gather rate constants and labels
        assert len(others) == len(labels), f"{labels} !~ {others}"
        all_ks = [self, *others]
        all_labels = [label, *labels]
        all_colors = COLOR_SEQUENCE[: len(all_labels)]

        # Gether data from functons
        T = numpy.linspace(*T_range, num=500)
        data_dct = {lab: k(T, P) for lab, k in zip(all_labels, all_ks, strict=True)}
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
            altair.Color(
                "key:N", scale=altair.Scale(domain=all_labels, range=all_colors)
            )
            if others
            else altair.value(all_colors[0])
        )

        # Create chart
        return (
            altair.Chart(data)
            .mark_line()
            .transform_fold(fold=list(data_dct.keys()))
            .encode(x=x, y=y, color=color)
        )


class RateConstant(BaseRateConstant):
    Ts: list[float]
    Ps: list[float]
    data: NDArray_

    # Private attributes
    _T_key = "T"
    _P_key = "P"
    type_: ClassVar[str] = "raw"
    _scalers: ClassVar[Scalers] = {"k_array": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "Ts": Dimension.temperature,
        "Ps": Dimension.pressure,
        "data": Dimension.rate_constant,
    }

    @property
    def kTP(self):  # noqa: N802
        return xarray.DataArray(
            data=self.data, coords={self._T_key: self.Ts, self._P_key: self.Ps}
        )

    @unit_.manage_units(
        [Dimension.temperature, Dimension.pressure], Dimension.rate_constant
    )
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate rate constant."""
        T_, P_ = self.process_input(T, P)
        kTP: NDArray[numpy.float64] = self.kTP.sel(T=T_, P=P_, method="ffill").data
        return self.process_output(kTP, T, P)


class RateConstantFit(BaseRateConstant):
    """Abstract base class for parametrized rate constants."""

    efficiencies: dict[str, float] = pydantic.Field(default_factory=dict)

    @property
    def third_body(self) -> str | None:
        """Get third body."""
        eff = self.efficiencies

        if eff.get("M") == 1:
            return "M"

        return next((c for c, e in eff.items() if e == 1.0), None)

    @pydantic.field_validator("efficiencies", mode="before")
    @classmethod
    def sanitize_efficiencies(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {k: v for k, v in value.items() if v is not None}
        return value


class ArrheniusRateConstantFit(RateConstantFit):
    A: float = 1.0
    b: float = 0.0
    E: float = 0.0

    # Private attributes
    type_: ClassVar[str] = "arrhenius"
    _scalers: ClassVar[Scalers] = {"A": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "A": Dimension.rate_constant,
        "E": Dimension.energy_per_substance,
    }

    @unit_.manage_units(
        [Dimension.temperature, Dimension.pressure], Dimension.rate_constant
    )
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate rate constant."""
        T_, P_ = self.process_input(T, P)
        R = unit_.system.gas_constant_value(UNITS)
        kTP = self.A * (T_**self.b) * numpy.exp(-self.E / (R * T_))
        return self.process_output(kTP, T, P)


class FalloffRateConstantFit(RateConstantFit, abc.ABC):  # type: ignore[misc]
    A_high: float
    b_high: float
    E_high: float
    A_low: float
    b_low: float
    E_low: float
    function: BlendingFunction_
    activated: bool = False

    # Private attributes
    type_: ClassVar[str] = "falloff"
    _scalers: ClassVar[Scalers] = {"A_high": numpy.multiply, "A_low": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "A_high": Dimension.rate_constant,
        "E_high": Dimension.energy_per_substance,
        "A_low": Dimension.rate_constant,
        "E_low": Dimension.energy_per_substance,
    }

    @unit_.manage_units(
        [Dimension.temperature, Dimension.pressure], Dimension.rate_constant
    )
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate rate constant."""
        T_, P_ = self.process_input(T, P)
        P_r = self.effective_reduced_pressure(T_, P_)
        if self.activated:
            k_low, _ = self.arrhenius_functions
            kTP = k_low(T_) / (1 + P_r) * self.function(T_, P_r)
        else:
            _, k_high = self.arrhenius_functions
            kTP = k_high(T_) * P_r / (1 + P_r) * self.function(T_, P_r)
        return self.process_output(kTP, T, P)

    @property
    def arrhenius_functions(
        self,
    ) -> tuple[ArrheniusRateConstantFit, ArrheniusRateConstantFit]:
        k_low = ArrheniusRateConstantFit(
            A=self.A_high, b=self.b_high, E=self.E_high, order=self.order
        )
        k_high = ArrheniusRateConstantFit(
            A=self.A_high, b=self.b_high, E=self.E_high, order=self.order
        )
        return k_low, k_high

    def effective_concentration(
        self,
        T: NDArray[numpy.float64],  # noqa: N803
        P: NDArray[numpy.float64],  # noqa: N803
    ) -> NDArray[numpy.float64]:
        """Get effective concentration(s) from temperature(s) and pressure(s).

        effective [M] = P / R T  (ideal gas law)

        :param T: Temperature(s)
        :param P: Pressure(s)
        :return: Effective concentration(s)
        """
        # Evaluate, using pint to handle units
        R_ = pint.Quantity("molar_gas_constant")
        T_ = pint.Quantity(T, UNITS.temperature)
        P_ = pint.Quantity(P, UNITS.pressure)
        m_ = P_ / (R_ * T_)

        # Return value in concentration units
        return m_.m_as(UNITS.concentration)

    def effective_reduced_pressure(
        self,
        T: NDArray[numpy.float64],  # noqa: N803
        P: NDArray[numpy.float64],  # noqa: N803
    ) -> NDArray[numpy.float64]:
        """Get effective concentration(s) from temperature(s) and pressure(s).

        effective P_r = k_low [M] / k_high  (ideal gas law)

        :param T: Temperature(s)
        :param P: Pressure(s)
        :return: Effective reduced pressure(s)
        """
        m_eff = self.effective_concentration(T, P)
        k_low, k_high = self.arrhenius_functions
        return k_low(T) * m_eff / k_high(T)


class PlogRateConstantFit(RateConstantFit):
    As: list[float]
    bs: list[float]
    Es: list[float]
    Ps: list[float]

    # Private attributes
    type_: ClassVar[str] = "plog"
    _scalers: ClassVar[Scalers] = {"As": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "As": Dimension.rate_constant,
        "Es": Dimension.energy_per_substance,
        "Ps": Dimension.pressure,
    }

    @unit_.manage_units(
        [Dimension.temperature, Dimension.pressure], Dimension.rate_constant
    )
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate rate constant for a single pressure."""
        T_, P_ = self.process_input(T, P)
        P0 = self.nearest_pressure(P_, which=0)
        P1 = self.nearest_pressure(P_, which=1)
        kT0 = self.nearest_arrhenius_values(T_, P_, which=0)
        kT1 = self.nearest_arrhenius_values(T_, P_, which=1)

        # Evaluate intermediate pressures
        log_P, log_P0, log_P1 = map(numpy.log, (P_, P0, P1))
        kTP = kT0 + (kT1 - kT0) * (log_P - log_P0) / (log_P1 - log_P0)

        # Evaluate on-boundary pressures (needed to fill in last pressure value)
        if numpy.ndim(P_) > 0:
            kTP[..., numpy.equal(P_, P0)] = kT0[..., numpy.equal(P_, P0)]
        elif P_ == P0:
            kTP = kT0

        return self.process_output(kTP, T, P)

    @property
    def arrhenius_functions(self) -> list[ArrheniusRateConstantFit]:
        return [
            ArrheniusRateConstantFit(A=A, b=b, E=E, order=self.order)
            for A, b, E in zip(self.As, self.bs, self.Es, strict=True)
        ]

    @property
    def pressures(self) -> NDArray[numpy.float64]:
        """Pressures."""
        return numpy.array(self.Ps, dtype=numpy.float64)

    @property
    def pressure_indices(self) -> list[int]:
        """Pressure indices."""
        return list(range(self.pressures.shape[0]))

    def nearest_arrhenius_values(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike,  # noqa: N803
        which: int = 0,
    ) -> NDArray[numpy.float64]:
        """Get nearest lower or higher pressure.

        :param P: Pressure(s)
        :param which: 0=lower, 1=higher
        :return: Nearest function(s)
        """
        iP = self.nearest_index(P, which=which)
        kTs = [k(T) for k in self.arrhenius_functions]
        return numpy.where(
            numpy.isin(iP, self.pressure_indices),
            numpy.choose(iP, kTs, mode="clip"),
            numpy.nan,
        )

    def nearest_pressure(
        self,
        P: ArrayLike,  # noqa: N803
        which: int = 0,
    ) -> NDArray[numpy.float64]:
        """Get nearest lower or higher pressure.

        :param P: Pressure(s)
        :param which: 0=lower, 1=higher
        :return: Nearest pressure(s)
        """
        iP = self.nearest_index(P, which=which)
        ps = self.pressures
        return numpy.where(
            numpy.isin(iP, self.pressure_indices),
            numpy.take(ps, iP, mode="clip"),
            numpy.nan,
        )

    def nearest_index(
        self,
        P: ArrayLike,  # noqa: N803
        which: int = 0,
    ) -> NDArray[numpy.int_]:
        """Get nearest lower or higher index.

        :param P: Pressure(s)
        :param which: 0=lower, 1=higher
        :return: Nearest index (indices)
        """
        return numpy.searchsorted(self.Ps, P, side="right") - 1 + which


class ChebRateConstantFit(RateConstantFit):
    coeffs: NDArray_
    T_range: tuple[float, float]
    P_range: tuple[float, float]

    # Private attributes
    type_: ClassVar[str] = "cheb"
    _scalers: ClassVar[Scalers] = {"coeffs": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "coeffs": Dimension.rate_constant,
        "T_range": Dimension.temperature,
        "P_range": Dimension.pressure,
    }

    @unit_.manage_units(
        [Dimension.temperature, Dimension.pressure], Dimension.rate_constant
    )
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate rate constant for a single pressure."""
        # Skip input processing, since chebgrid2d automatically forms the grid
        T0, T1 = self.T_range
        P0, P1 = self.P_range

        inv_ = numpy.reciprocal
        log_ = numpy.log10

        T_r = (2 * inv_(T) - inv_(T0) - inv_(T1)) / (inv_(T1) - inv_(T0))
        P_r = (2 * log_(P) - log_(P0) - log_(P1)) / (log_(P1) - log_(P0))

        # AVC: I don't understand why I need to transpose the coefficient matrix here
        kTP = chebyshev.chebgrid2d(T_r, P_r, self.coeffs.T)
        return self.process_output(kTP, T, P)


RateConstant_ = Annotated[
    pydantic.SkipValidation[BaseRateConstant],
    pydantic.BeforeValidator(lambda x: BaseRateConstant.model_validate(x)),
    pydantic.PlainSerializer(lambda x: BaseRateConstant.model_validate(x).model_dump()),
    pydantic.GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(pydantic.BaseModel))
    ),
]


# Conversions
def chemkin_string(rate_const: RateConstantFit, eq_width: int = 0) -> str:
    """Write Chemkin rate to a string.

    :param rate_const: Rate constant
    :param eq_width: Equation width for alignment
    :return: Chemkin rate string
    """
    # Generate dummy head line
    head_line = chemkin.write_numbers([1.0, 0.0, 0.0])

    # Calculate the total width of the top line for alignment
    head_width = eq_width + len(head_line) + 1

    # Generate auxiliary lines and replace head line, if appropriate
    match rate_const:
        case ArrheniusRateConstantFit():
            head_params = [rate_const.A, rate_const.b, rate_const.E]
            head_line = chemkin.write_numbers(head_params)
            aux_lines = []
        case FalloffRateConstantFit(activated=False):
            high_params = [rate_const.A_high, rate_const.b_high, rate_const.E_high]
            low_params = [rate_const.A_low, rate_const.b_low, rate_const.E_low]
            head_line = chemkin.write_numbers(high_params)
            aux_lines = [chemkin.write_aux("LOW", low_params, head_width=head_width)]
        case FalloffRateConstantFit(activated=True):
            high_params = [rate_const.A_high, rate_const.b_high, rate_const.E_high]
            low_params = [rate_const.A_low, rate_const.b_low, rate_const.E_low]
            head_line = chemkin.write_numbers(low_params)
            aux_lines = [chemkin.write_aux("HIGH", high_params, head_width=head_width)]
        case PlogRateConstantFit():
            plog_params = [rate_const.Ps, rate_const.As, rate_const.bs, rate_const.Es]
            aux_lines = [
                chemkin.write_aux("PLOG", row) for row in zip(*plog_params, strict=True)
            ]
        case ChebRateConstantFit():
            shape = numpy.shape(rate_const.coeffs)
            aux_lines = [
                chemkin.write_aux("TCHEB", rate_const.T_range, head_width=head_width),
                chemkin.write_aux("PCHEB", rate_const.P_range, head_width=head_width),
                chemkin.write_aux("CHEB", shape, head_width=head_width, as_int=True),
                *(
                    chemkin.write_aux("CHEB", row, head_width=head_width)
                    for row in rate_const.coeffs.tolist()
                ),
            ]
        case _:
            raise ValueError(
                f"Rate constant has unknown type {type(rate_const)}:\n{rate_const}"
            )

    if isinstance(rate_const, FalloffRateConstantFit):
        aux_lines.extend(
            blending_function_chemkin_aux_lines(
                rate_const.function, head_width=head_width
            )
        )

    eff_line = chemkin.write_efficiencies(
        rate_const.efficiencies, third_body=rate_const.third_body
    )
    if eff_line is not None:
        aux_lines.append(eff_line)

    lines = [head_line, *aux_lines]
    return "\n".join(lines)


# Parse helpers
def extract_rate_constant_from_chemkin_parse_results(
    res: chemkin.ChemkinRateParseResults, units: UnitsData | None = None
) -> RateConstantFit:
    """Extract rate data from Chemkin parse results.

    Chemkin parse results are modified in-place

    :param res: Chemkin rate parse results
    :return: Rate data
    """
    # Extract efficiencies
    efficiencies = res.efficiencies.copy()
    res.efficiencies.clear()

    # Determine reaction order
    order = len(res.reactants) + bool(efficiencies)

    # Determine units
    units = Units() if units is None else Units.model_validate(units)

    if "CHEB" in res.aux_numbers:
        # Read coefficients
        cheb: list[float] = res.aux_numbers.pop("CHEB")
        shape = tuple(map(int, cheb[:2]))
        coeffs = numpy.reshape(cheb[2:], shape)
        # Read ranges
        t_range = res.aux_numbers.pop("TCHEB")
        p_range = res.aux_numbers.pop("PCHEB")
        return ChebRateConstantFit(
            coeffs=coeffs,
            T_range=t_range,
            P_range=p_range,
            efficiencies=efficiencies,
            order=order,
            units=units,
        )

    if "PLOG" in res.aux_numbers:
        ps, As, bs, Es = zip(
            *mit.chunked(res.aux_numbers.pop("PLOG"), 4, strict=True), strict=True
        )
        return PlogRateConstantFit(
            As=As,
            bs=bs,
            Es=Es,
            Ps=ps,
            efficiencies=efficiencies,
            order=order,
            units=units,
        )

    if "LOW" in res.aux_numbers:
        A_high, b_high, E_high = res.arrhenius
        A_low, b_low, E_low = res.aux_numbers.pop("LOW")
        function = extract_blending_function_from_chemkin_parse_results(res)
        return FalloffRateConstantFit(
            A_low=A_low,
            b_low=b_low,
            E_low=E_low,
            A_high=A_high,
            b_high=b_high,
            E_high=E_high,
            function=function,
            activated=False,
            efficiencies=efficiencies,
            order=order,
            units=units,
        )

    if "HIGH" in res.aux_numbers:
        A_low, b_low, E_low = res.arrhenius
        A_high, b_high, E_high = res.aux_numbers.pop("HIGH")
        function = extract_blending_function_from_chemkin_parse_results(res)
        return FalloffRateConstantFit(
            A_low=A_low,
            b_low=b_low,
            E_low=E_low,
            A_high=A_high,
            b_high=b_high,
            E_high=E_high,
            function=function,
            activated=True,
            efficiencies=efficiencies,
            order=order,
            units=units,
        )

    A, b, E = res.arrhenius
    return ArrheniusRateConstantFit(
        A=A, b=b, E=E, efficiencies=efficiencies, order=order, units=units
    )
