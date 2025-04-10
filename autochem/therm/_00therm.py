"""Thermodynamic data."""

import abc
import itertools
from typing import ClassVar

import numpy
import pint
from numpy.typing import NDArray

from .. import unit_
from ..unit_ import UNITS, Const, Dim, Dimension, UnitManager, Units, UnitsData, system
from ..util import chemkin
from ..util.type_ import Frozen, Scalable, Scalers, SubclassTyped


class ThermBase(UnitManager, Frozen, SubclassTyped, abc.ABC):
    """Abstract base class for thermodynamic data."""


class ThermData(ThermBase):
    """Raw thermodynamic data.

    :param Ts: Temperatures (K)
    :param Z0s: Logs of one-particle partition function per volume, ln(Q_1' [cm^-3])
    :param Z1s: First temperature derivatives of Z0, d(ln(Q_1'))/dT [K^-1]
    :param Z2s: Second temperature derivatives of Z0, d^2(ln(Q_1'))/dT^2 [K^-2]
    """

    Ts: list[float]
    Z0s: list[float]
    Z1s: list[float]
    Z2s: list[float]

    # Private attributes
    type_: ClassVar[str] = "data"
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "Ts": Dim.temperature,
        "Z0s": system.log(Dim.volume),
        "Z1s": Dim.temperature**-1,
        "Z2s": Dim.temperature**-2,
    }

    # TODO: Fix unit handling here
    @unit_.manage_units([], Dim.energy_per_substance)
    def enthalpy(
        self,
        H0: float,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Calculate enthalpy.

        Formula:

            H = H_0 + R (T^2 d(ln(Q_1'))/dT + T)
              = H_0 + R (T^2 Z_1 + T)

        :param H0: Reference enthalpy
        :param units: Units
        :return: Enthalpy
        """
        # Evaluate
        R = unit_.system.constant_value(Const.gas, UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z1 = numpy.array(self.Z1s, dtype=numpy.float64)
        H = H0 + R * (T**2 * Z1 + T)
        return H

    @unit_.manage_units([], Dim.energy_per_substance / Dim.temperature)
    def heat_capacity(
        self,
        const_P: bool = False,  # noqa: N803
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
        R = unit_.system.constant_value(Const.gas, UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z1 = numpy.array(self.Z1s, dtype=numpy.float64)
        Z2 = numpy.array(self.Z2s, dtype=numpy.float64)
        C = R * (2 * T * Z1 + T**2 * Z2)
        C += R if const_P else 0.0
        return C

    @unit_.manage_units([], Dim.energy_per_substance / Dim.temperature)
    def entropy(
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
        R = unit_.system.constant_value(Const.gas, UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z0 = numpy.array(self.Z0s, dtype=numpy.float64)
        Z1 = numpy.array(self.Z1s, dtype=numpy.float64)
        S = R * (T * Z1 + Z0 - numpy.log(c) + 1)
        return S


class ThermFit(ThermBase, Scalable, abc.ABC):
    """Fitted thermodynamic data."""

    T_low: float
    T_high: float


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


def extract_thermo_data_from_messpf_string(pf_str: str) -> ThermData:
    """Extract thermo data from MESS-PF output string.

    :param pf_str: MESS-PF output string
    :return: Thermo data
    """
    lines = list(map(str.strip, pf_str.strip().splitlines()))
    lines = list(itertools.dropwhile(lambda s: not s.startswith("Z_0"), lines))
    data = [list(map(float, line.split())) for line in lines[1:]]
    Ts, Z0s, Z1s, Z2s, *_ = map(list, zip(*data, strict=True))
    return ThermData(Ts=Ts, Z0s=Z0s, Z1s=Z1s, Z2s=Z2s)


def extract_thermo_from_chemkin_parse_results(
    res: chemkin.ChemkinThermoParseResults,
    T_mid: float = 1000,  # noqa: N803
) -> ThermFit:
    """Extract thermo data from Chemkin parse results.

    :param res: Chemkin thermo parse results
    :param T_mid: Default mid-point temperature
    :return: Thermo data fit
    """
    if len(res.coeffs) == 14:
        return Nasa7ThermFit(
            T_low=res.T_low,
            T_high=res.T_high,
            coeffs_low=res.coeffs[7:],
            coeffs_high=res.coeffs[:7],
            T_mid=res.T_mid or T_mid,
        )

    raise ValueError(f"Unable to interpret parse results: {res}")
