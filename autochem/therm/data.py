"""Thermodynamic data."""

import abc
import itertools
from typing import Annotated, ClassVar, Literal

import numpy
import pint
import pydantic
import xarray
from numpy.typing import ArrayLike, NDArray
from pydantic_core import core_schema

from .. import unit_
from ..unit_ import UNITS, C, D, Dimension, UnitManager, Units, UnitsData, dim
from ..util import FormulaData, chemkin, form, pac99
from ..util.type_ import Frozen, Scalable, Scalers, SubclassTyped
from .func import Bounded, Nasa7Calculator, ThermCalculator


class BaseTherm(ThermCalculator, UnitManager, Frozen, SubclassTyped, abc.ABC):
    """Abstract base class for thermodynamic data."""

    formula: dict[str, int]
    charge: int = 0


class Therm(BaseTherm):
    """Raw thermodynamic data.

    :param Ts: Temperatures (K)
    :param Z0s: Logs of one-particle partition function per volume, ln(Q_1' [cm^-3])
    :param Z1s: First temperature derivatives of Z0, d(ln(Q_1'))/dT [K^-1]
    :param Z2s: Second temperature derivatives of Z0, d^2(ln(Q_1'))/dT^2 [K^-2]
    :param Hf: Enthalpy of formation at 298.15 K [cal/mol]
    """

    Ts: list[float]
    Z0s: list[float]
    Z1s: list[float]
    Z2s: list[float]
    Hf: float | None = None

    # Private attributes
    type_: ClassVar[str] = "data"
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "Ts": D.temperature,
        "Z0s": dim.log(D.volume),
        "Z1s": D.temperature**-1,
        "Z2s": D.temperature**-2,
    }

    @pydantic.model_validator(mode="after")
    def sort_by_temperature(self):
        """Sort arrays by temperature."""
        frozen = self.model_config["frozen"]
        self.model_config["frozen"] = False
        self.Ts, self.Z0s, self.Z1s, self.Z2s = map(
            list,
            zip(
                *sorted(zip(self.Ts, self.Z0s, self.Z1s, self.Z2s, strict=True)),
                strict=True,
            ),
        )
        self.model_config["frozen"] = frozen
        return self

    # Replace this with a model validator (before or after), allowing extra arguments
    def __init__(
        self,
        formula: dict[str, int],
        Hf: float | None = None,  # noqa: N803
        Tf: float = 298.15,  # noqa: N803
        **kwargs,
    ):
        """Initialize, handling enthalpy of formation.

        :param Hf: Enthalpy of formation at 0 K or 298.15 K
        :param Tf: Reference temperature for enthalpy of formation, 0 K or 298.15 K
        """
        super().__init__(formula=formula, Hf=Hf, **kwargs)

        if int(Tf) == 0:
            frozen = self.model_config["frozen"]
            self.model_config["frozen"] = False
            self.Hf += self.delta_enthalpy(
                T=298, method="nearest"
            ) - elemental_delta_enthalpy_room_temperature(self.formula)
            self.model_config["frozen"] = frozen
        elif not int(Tf) == 298:
            raise ValueError(f"Invalid reference temperature Tf = {Tf}")

    # Properties
    @unit_.manage_units([], D.energy_per_substance)
    def enthalpy_of_formation(self, units: UnitsData | None = None) -> float:
        """Get enthalpy of formation at 298.15 K."""
        assert self.Hf is not None, "Enthalpy of formation not set"
        return self.Hf

    @property
    def data_set(self) -> xarray.Dataset:
        """Access data as an xarray Dataset."""
        coord_key = "T"
        coord_vals = self.Ts
        data_arrs = {
            "Z0": self.Z0s,
            "Z1": self.Z1s,
            "Z2": self.Z2s,
            "dH": self.delta_enthalpy_data(),
            "S": self.entropy_data(),
            "Cv": self.heat_capacity_data(const="V"),
            "Cp": self.heat_capacity_data(const="P"),
        }
        return xarray.Dataset(
            data_vars={k: ([coord_key], v) for k, v in data_arrs.items()},
            coords={coord_key: coord_vals},
        )

    # Thermodynamic function data points
    @unit_.manage_units([], D.energy_per_substance)
    def delta_enthalpy_data(
        self, units: UnitsData | None = None
    ) -> NDArray[numpy.float64]:
        """Calculate delta enthalpy data.

        Formula:

            dH = R (T^2 d(ln(Q_1'))/dT + T)
               = R (T^2 Z_1 + T)

        :param units: Units
        :return: Enthalpy
        """
        # Evaluate
        R = unit_.const.value(C.gas, UNITS)
        Ts = numpy.array(self.Ts, dtype=numpy.float64)
        Z1s = numpy.array(self.Z1s, dtype=numpy.float64)
        Hs = R * (Ts**2 * Z1s + Ts)
        return Hs

    @unit_.manage_units([], D.energy_per_substance)
    def enthalpy_data(
        self, units: UnitsData | None = None
    ) -> NDArray[numpy.float64]:
        """Calculate enthalpy data.

        Formula:

            H(T) = H0 + dH(T)
            H0 = Hf(298.15 K) - dH(298.15 K)

        :param units: Units
        :return: Enthalpy
        """
        # Evaluate
        R = unit_.const.value(C.gas, UNITS)
        Ts = numpy.array(self.Ts, dtype=numpy.float64)
        Z1s = numpy.array(self.Z1s, dtype=numpy.float64)
        Hs = R * (Ts**2 * Z1s + Ts)
        return Hs

    # TODO: Fix unit manager to handle pressure keyword input
    @unit_.manage_units([], D.energy_per_substance / D.temperature)
    def entropy_data(
        self,
        P: float = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Calculatate entropy.

        Formula:

            S = R (T d(ln(Q_1'))/dT + ln(Q_1') - ln(c) + 1)
              = R (T Z1 + Z0 - ln(c) + 1)

        in terms of the one-particle partition function per unit volume, Q_1', and the
        standard concentration, c = N_A / V, which comes from accounting for volume.
        The latter is determined for a standard state pressure via the ideal gas law:

           c = P / k_B T

        This must be converted to internal units.

        :param P: Pressure
        :param units: Units
        :return: Entropy
        """
        T = numpy.array(self.Ts, dtype=numpy.float64)

        # Evaluate standard concentration (molecules* / volume) from pressure  *implicit
        units = Units.model_validate(units) if units is not None else UNITS
        k_B_ = pint.Quantity("boltzmann_constant")
        T_ = pint.Quantity(T, UNITS.temperature)
        P_ = pint.Quantity(P, units.pressure)
        c = (P_ / (k_B_ * T_)).m_as("1/cm**3")

        # Evaluate
        R = unit_.const.value(C.gas, UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z0 = numpy.array(self.Z0s, dtype=numpy.float64)
        Z1 = numpy.array(self.Z1s, dtype=numpy.float64)
        S = R * (T * Z1 + Z0 - numpy.log(c) + 1)
        return S

    @unit_.manage_units([], D.energy_per_substance / D.temperature)
    def heat_capacity_data(
        self,
        const: Literal["P", "V"] = "P",
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Calculate the heat capacity at constant volume or pressure.

        Formula:

            C_v = R (2 T d(ln(Q_1'))/dT + T^2 d^2(ln(Q_1'))/dT^2)
                = R (2 T Z_1 + T^2 Z_2)
            C_p = C_v + R = R (1 + 2 T Z_1 + T^2 Z_2)

        :param const: Whether to hold pressure ("P") or volume ("V") constant
        :param units: Units
        :return: Heat capacity
        """
        # Evaluate
        R = unit_.const.value(C.gas, UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z1 = numpy.array(self.Z1s, dtype=numpy.float64)
        Z2 = numpy.array(self.Z2s, dtype=numpy.float64)
        C_ = R * (1 + 2 * T * Z1 + T**2 * Z2)
        C_ -= R if const == "V" else 0.0
        return C_

    # Thermodynamic function calculators
    @unit_.manage_units([], D.energy_per_substance)
    def delta_enthalpy(
        self,
        T: ArrayLike,  # noqa: N803
        method: str = "nearest",
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Calculate enthalpy.

        Formula:

            dH = R (T^2 d(ln(Q_1'))/dT + T)
               = R (T^2 Z_1 + T)

        :param T: Temperature for evaluation
        :param method: Xarray data selection method
        :param units: Units
        :return: Enthalpy
        """
        return self.data_set["dH"].sel(T=T, method=method).data

    def heat_capacity(
        self,
        T: ArrayLike,  # noqa: N803
        const: Literal["P", "V"] = "P",
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate heat capacity, C_V(T) or C_P(T).

        :param T: Temperature(s)
        :param const: Whether to hold pressure ("P") or volume ("V") constant
        :param units: Unit system
        :return: Function value(s)
        """
        pass

    def enthalpy(
        self,
        T: ArrayLike,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate enthalpy, H(T).

        :param T: Temperature(s)
        :param units: Unit system
        :return: Function value(s)
        """
        pass

    def entropy(
        self,
        T: ArrayLike,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate entropy, S(T).

        :param T: Temperature(s)
        :param units: Unit system
        :return: Function value(s)
        """
        pass


class ThermFit(BaseTherm, Bounded, Scalable, abc.ABC):
    """Fitted thermodynamic data."""

    pass


class Nasa7ThermFit(ThermFit, ThermCalculator):
    """Fitted thermodynamic data."""

    T_mid: float
    coeffs_low: list[float]
    coeffs_high: list[float]

    # Private attributes
    type_: ClassVar[str] = "nasa7"
    _scalers: ClassVar[Scalers] = {
        "coeffs_low": numpy.multiply,
        "coeffs_high": numpy.multiply,
    }

    def piecewise_calculators(self) -> list[Nasa7Calculator]:
        """Get calculators for a numpy.piecewise evaluation.

        :return: Pair of calculators
        """
        coeff_keys = ("a0", "a1", "a2", "a3", "a4", "a5", "a6")
        calc_low = Nasa7Calculator(
            T_min=self.T_min,
            T_max=self.T_mid,
            **dict(zip(coeff_keys, self.coeffs_low, strict=True)),
        )
        calc_high = Nasa7Calculator(
            T_min=self.T_mid,
            T_max=self.T_max,
            **dict(zip(coeff_keys, self.coeffs_high, strict=True)),
        )
        return [calc_low, calc_high]

    def piecewise_conditions(
        self,
        T: ArrayLike,  # noqa: N803
    ) -> list[NDArray[numpy.bool_]]:
        """Get conditions for a numpy.piecewise evaluation.

        :param T: Temperature(s)
        :return: Pair of boolean arrays
        """
        calc_low, calc_high = self.piecewise_calculators()
        return [
            calc_low.in_bounds(T, include_max=False),
            calc_high.in_bounds(T, include_max=True),
        ]

    def heat_capacity(
        self,
        T: ArrayLike,  # noqa: N803
        const: Literal["P", "V"] = "P",
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate heat capacity, C_V(T) or C_P(T).

        :param T: Temperature(s)
        :param const: Whether to hold pressure ("P") or volume ("V") constant
        :param units: Unit system
        :return: Function value(s)
        """
        conds = self.piecewise_conditions(T)
        funcs = [calc.heat_capacity for calc in self.piecewise_calculators()]
        return numpy.piecewise(T, conds, funcs)

    def enthalpy(
        self,
        T: ArrayLike,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate enthalpy, H(T).

        :param T: Temperature(s)
        :param units: Unit system
        :return: Function value(s)
        """
        conds = self.piecewise_conditions(T)
        funcs = [calc.enthalpy for calc in self.piecewise_calculators()]
        return numpy.piecewise(T, conds, funcs)

    def entropy(
        self,
        T: ArrayLike,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate entropy, S(T).

        :param T: Temperature(s)
        :param units: Unit system
        :return: Function value(s)
        """
        conds = self.piecewise_conditions(T)
        funcs = [calc.entropy for calc in self.piecewise_calculators()]
        return numpy.piecewise(T, conds, funcs)


Therm_ = Annotated[
    pydantic.SkipValidation[BaseTherm],
    pydantic.BeforeValidator(lambda x: BaseTherm.model_validate(x)),
    pydantic.PlainSerializer(lambda x: BaseTherm.model_validate(x).model_dump()),
    pydantic.GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(pydantic.BaseModel))
    ),
]


def from_messpf_output_string(
    pf_str: str,
    formula: FormulaData,
    charge: int = 0,
    Hf: float | None = None,  # noqa: N803
    Tf: float = 0,  # noqa: N803
    units: UnitsData | None = None,
) -> Therm:
    """Extract thermo data from MESS-PF output string.

    :param pf_str: MESS-PF output string
    :param formula: Formula
    :param charge: Molecular charge
    :param Hf: Enthalpy of formation at 0 K or 298 K
    :param Tf: Enthalpy of formation reference temperature, 0 K or 298 K
    :param units: Units of enthalpy of formation
    :return: Thermo data
    """
    units0 = UNITS if units is None else Units.model_validate(units)
    Hf = None if Hf is None else dim.convert(units0, UNITS, D.energy, Hf)
    formula = form.normalize_input(formula)
    lines = list(map(str.strip, pf_str.strip().splitlines()))
    lines = list(itertools.dropwhile(lambda s: not s.startswith("Z_0"), lines))
    data = [list(map(float, line.split())) for line in lines[1:]]
    Ts, Z0s, Z1s, Z2s, *_ = map(list, zip(*data, strict=True))
    # Replace 298.2 with 298.15 (needed for PAC99 input to run)
    Ts = [298.15 if round(T) == 298 else T for T in Ts]
    return Therm(
        Ts=Ts, Z0s=Z0s, Z1s=Z1s, Z2s=Z2s, Hf=Hf, Tf=Tf, formula=formula, charge=charge
    )


def from_chemkin_string(
    therm_str: str,
    T_mid: float = 1000,  # noqa: N803
) -> ThermFit:
    """Read species thermo from Chemkin string.

    :param therm_str: Chemkin species therm string
    :param T_mid: Default mid-point temperature
    :return: Species thermo
    """
    # Parse string
    res = chemkin.parse_thermo(therm_str)

    # Extract thermo data
    return from_chemkin_parse_results(res, T_mid=T_mid)


def from_chemkin_parse_results(
    res: chemkin.ChemkinThermoParseResults,
    T_mid: float = 1000,  # noqa: N803
) -> ThermFit:
    """Extract thermo data from Chemkin parse results.

    :param res: Chemkin thermo parse results
    :param T_mid: Default mid-point temperature
    :return: Thermo data fit
    """
    # Determine charge, if any
    charge = 0
    formula = res.formula.copy()
    if "E" in formula:
        charge = -formula.pop("E")

    # Read in coefficients
    if len(res.coeffs) == 14:
        return Nasa7ThermFit(
            T_min=res.T_min,
            T_max=res.T_max,
            coeffs_low=res.coeffs[7:],
            coeffs_high=res.coeffs[:7],
            T_mid=res.T_mid or T_mid,
            formula=formula,
            charge=charge,
        )

    raise ValueError(f"Unable to interpret parse results: {res}")


def from_pac99_output_string(therm_str: str) -> Nasa7ThermFit:
    """Read species thermo from PAC99 output string.

    :param therm_str: PAC99 .c97 output string
    :return: NASA7 thermo fit
    """
    # Parse string
    res = pac99.parse_thermo(therm_str)

    # Extract thermo data
    return from_pac99_output_parse_results(res)


def from_pac99_output_parse_results(
    res: pac99.Pac99ThermoParseResults,
) -> Nasa7ThermFit:
    """Extract thermo data from Chemkin parse results.

    :param res: Chemkin thermo parse results
    :param T_mid: Default mid-point temperature
    :return: Thermo data fit
    """
    # Determine charge, if any
    charge = 0
    formula = res.formula.copy()
    if "E" in formula:
        charge = -formula.pop("E")

    assert len(res.ranges) == len(res.coeffs_lst) == 2, res

    coeffs_low, coeffs_high = res.coeffs_lst
    (T_min, T_mid), (T_mid_, T_max) = res.ranges
    assert T_mid == T_mid_, f"{T_mid} != {T_mid_}"

    # Read in coefficients
    return Nasa7ThermFit(
        T_min=T_min,
        T_max=T_max,
        coeffs_low=coeffs_low,
        coeffs_high=coeffs_high,
        T_mid=T_mid,
        formula=formula,
        charge=charge,
    )


# Helpers
KJ_TO_CAL = pint.Quantity(1, "kJ").m_as("cal")
ENTHALPY_CHANGE_0K_TO_298K: dict[str, float] = {
    # Monatomic values (kJ)
    "He": 6.197,
    "Ne": 6.197,
    "Ar": 6.197,
    "Kr": 6.197,
    "Xe": 6.197,
    "S": 4.412,
    "P": 5.36,
    "C": 1.05,
    "Si": 3.217,
    "Ge": 4.636,
    "Sn": 6.323,
    "Pb": 6.87,
    "B": 1.222,
    "Al": 4.54,
    "Zn": 5.647,
    "Cd": 6.247,
    "Hg": 9.342,
    "Cu": 5.004,
    "Ag": 5.745,
    "Ti": 4.824,
    "U": 6.364,
    "Th": 6.35,
    "Be": 1.95,
    "Mg": 4.998,
    "Ca": 5.736,
    "Li": 4.632,
    "Na": 6.46,
    "K": 7.088,
    "Rb": 7.489,
    "Cs": 7.711,
    # Diatomic values (kJ)
    "O": 8.68 / 2,
    "H": 8.468 / 2,
    "F": 8.825 / 2,
    "Cl": 9.181 / 2,
    "Br": 24.52 / 2,
    "N": 8.67 / 2,
}
ENTHALPY_CHANGE_0K_TO_298K = {
    k: v * KJ_TO_CAL for k, v in ENTHALPY_CHANGE_0K_TO_298K.items()
}


def elemental_delta_enthalpy_room_temperature(
    formula: FormulaData,
) -> float:
    """Get change in enthalpy of formation for increasing from 0 K to room temperature
    (298 K).

    :param formula: Formula
    """
    formula = form.normalize_input(formula)
    assert all(k in ENTHALPY_CHANGE_0K_TO_298K for k in formula), (
        f"Invalid symbols: {formula}"
    )
    return sum(ENTHALPY_CHANGE_0K_TO_298K[k] * v for k, v in formula.items())
