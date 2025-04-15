"""Thermodynamic data."""

import abc
import itertools
from typing import Annotated, ClassVar

import numpy
import pint
import pydantic
import xarray
from numpy.typing import ArrayLike, NDArray
from pydantic_core import core_schema

from .. import unit_
from ..unit_ import UNITS, C, D, Dimension, UnitManager, Units, UnitsData, const, dim
from ..util import FormulaData, chemkin, form
from ..util.type_ import Frozen, Scalable, Scalers, SubclassTyped


class BaseTherm(UnitManager, Frozen, SubclassTyped, abc.ABC):
    """Abstract base class for thermodynamic data."""

    formula: dict[str, int]
    charge: int = 0


class Therm(BaseTherm):
    """Raw thermodynamic data.

    :param Ts: Temperatures (K)
    :param Z0s: Logs of one-particle partition function per volume, ln(Q_1' [cm^-3])
    :param Z1s: First temperature derivatives of Z0, d(ln(Q_1'))/dT [K^-1]
    :param Z2s: Second temperature derivatives of Z0, d^2(ln(Q_1'))/dT^2 [K^-2]
    :param Hf: Enthalpy of formation at 0 K [cal/mol]
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
        Tf: float = 0,  # noqa: N803
        **kwargs,
    ):
        """Initialize, handling enthalpy of formation.

        :param Hf: Enthalpy of formation at 0 K or 298 K
        :param Tf: Reference temperature for enthalpy of formation, 0 K or 298 K
        """
        super().__init__(formula=formula, Hf=Hf, **kwargs)

        if int(Tf) == 298:
            frozen = self.model_config["frozen"]
            self.model_config["frozen"] = False
            self.Hf -= self.delta_enthalpy_of_formation_room_temperature()
            self.model_config["frozen"] = frozen
        elif not int(Tf) == 0:
            raise ValueError(f"Invalid reference temperature Tf = {Tf}")

    @property
    def data_set(self) -> xarray.Dataset:
        """Access data as an xarray Dataset."""
        coord_key = "T"
        coord_vals = self.Ts
        data_arrs = {
            "Z0": self.Z0s,
            "Z1": self.Z1s,
            "Z2": self.Z2s,
            "H": self.delta_enthalpy_data(),
            "S": self.entropy_data(),
            "Cv": self.heat_capacity_data(at_const_P=False),
            "Cp": self.heat_capacity_data(at_const_P=True),
        }
        return xarray.Dataset(
            data_vars={k: ([coord_key], v) for k, v in data_arrs.items()},
            coords={coord_key: coord_vals},
        )

    @unit_.manage_units([], D.energy_per_substance)
    def delta_enthalpy_data(
        self, units: UnitsData | None = None
    ) -> NDArray[numpy.float64]:
        """Calculate delta enthalpy data.

        Formula:

            dH = R (T^2 d(ln(Q_1'))/dT + T)
               = R (T^2 Z_1 + T)

        :param T: Temperature for evaluation
        :param H0: Reference enthalpy
        :param units: Units
        :return: Enthalpy
        """
        # Evaluate
        R = const.value(C.gas, UNITS)
        Ts = numpy.array(self.Ts, dtype=numpy.float64)
        Z1s = numpy.array(self.Z1s, dtype=numpy.float64)
        Hs = R * (Ts**2 * Z1s + Ts)
        return Hs

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
        return self.data_set["H"].sel(T=T, method=method).data

    @unit_.manage_units([], D.energy_per_substance / D.temperature)
    def heat_capacity_data(
        self,
        at_const_P: bool = False,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Calculate the heat capacity at constant volume or pressure.

        Formula:

            C_v = R (2 T d(ln(Q_1'))/dT + T^2 d^2(ln(Q_1'))/dT^2)
                = R (2 T Z_1 + T^2 Z_2)
            C_p = C_v + R = R (1 + 2 T Z_1 + T^2 Z_2)

        :param units: Units
        :param const_P: Calculate at constant pressure? Otherwise, constant volume.
        :return: Heat capacity
        """
        # Evaluate
        R = const.value(C.gas, UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z1 = numpy.array(self.Z1s, dtype=numpy.float64)
        Z2 = numpy.array(self.Z2s, dtype=numpy.float64)
        heat_capacity = R * (2 * T * Z1 + T**2 * Z2)
        heat_capacity += R if at_const_P else 0.0
        return heat_capacity

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

        in terms of the one-particle partition function per unit volume, Q_1',
        and the standard concentration, c, N_A / V, which comes from accounting for
        volume. The latter via the ideal gas law by a standard state pressure:

           c = P / k_B T

        This must be converted to interna

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
        R = const.value(C.gas, UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z0 = numpy.array(self.Z0s, dtype=numpy.float64)
        Z1 = numpy.array(self.Z1s, dtype=numpy.float64)
        entropy = R * (T * Z1 + Z0 - numpy.log(c) + 1)
        return entropy

    def delta_enthalpy_of_formation_room_temperature(self):
        """Calculate enthalpy difference from zero to room temperature."""
        return self.delta_enthalpy(
            T=298, method="nearest"
        ) - elemental_delta_enthalpy_room_temperature(self.formula)


class ThermFit(BaseTherm, Scalable, abc.ABC):
    """Fitted thermodynamic data."""

    T_low: float
    T_high: float


class ShomateFit(ThermFit):
    """Fitted thermodynamic data."""

    T_mid: float
    coeffs_low: list[float]
    coeffs_high: list[float]

    # Private attributes
    type_: ClassVar[str] = "shomate"
    _scalers: ClassVar[Scalers] = {
        "coeffs_low": numpy.multiply,
        "coeffs_high": numpy.multiply,
    }


class Nasa7ThermFit(ThermFit):
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
    return Therm(
        Ts=Ts, Z0s=Z0s, Z1s=Z1s, Z2s=Z2s, Hf=Hf, Tf=Tf, formula=formula, charge=charge
    )


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
        charge = formula.pop("E")

    # Read in coefficients
    if len(res.coeffs) == 14:
        return Nasa7ThermFit(
            T_low=res.T_low,
            T_high=res.T_high,
            coeffs_low=res.coeffs[7:],
            coeffs_high=res.coeffs[:7],
            T_mid=res.T_mid or T_mid,
            formula=formula,
            charge=charge,
        )

    raise ValueError(f"Unable to interpret parse results: {res}")


# Helpers
KJ_TO_CAL = pint.Quantity(1, "kJ").m_as("cal")
ENTHALPY_CHANGE_0K_TO_298K = {
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
    return sum(ENTHALPY_CHANGE_0K_TO_298K.get(k) * v for k, v in formula.items())
