"""Thermodynamic data."""

import abc
import itertools
from typing import ClassVar

import numpy
from numpy.typing import NDArray

from .. import unit_
from ..unit_ import UNITS, Dim, Units, UnitsData
from ..util import chemkin
from ..util.type_ import Frozen, Scalable, Scalers, SubclassTyped


class ThermBase(Frozen, Scalable, SubclassTyped, abc.ABC):
    """Abstract base class for thermodynamic data."""


class ThermData(ThermBase):
    """Raw thermodynamic data.

    :param Ts: Temperatures (K)
    :param Z0s: Partition function natural logarithm, ln(Q)
    :param Z1s: First derivative, d(ln(Q))/dT  (derivative w.r.t beta?? or T in energy units?)
    :param Z2s: Second derivative, d^2(ln(Q))/dT^2
    """

    Ts: list[float]
    Z0s: list[float]
    Z1s: list[float]
    Z2s: list[float]

    # Private attributes
    type_: ClassVar[str] = "data"

    @unit_.manage_units([], Dim.energy_per_substance)
    def internal_energy(self, units: UnitsData | None = None) -> NDArray[numpy.float64]:
        """Calculate internal energy.

        U = <E> = R T^2 d(ln(Q))/dT

        :param units: Units
        :return: Internal energy
        """
        # Evaluate
        R = unit_.system.gas_constant_value(UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z1 = numpy.array(self.Z1s, dtype=numpy.float64)
        U = R * T**2 * Z1
        return U

    @unit_.manage_units([], Dim.energy_per_substance / Dim.temperature)
    def entropy(self, units: UnitsData | None = None):
        """Calculatate entropy.

        S = <E> / T + R ln(Q)

        :param units: Units
        :return: Entropy
        """
        # Evaluate
        U = self.internal_energy(units=UNITS)
        R = unit_.system.gas_constant_value(UNITS)
        T = numpy.array(self.Ts, dtype=numpy.float64)
        Z0 = numpy.array(self.Z0s, dtype=numpy.float64)
        S = U / T + R * Z0
        print(S)
        return S


class ThermFit(ThermBase, abc.ABC):
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
    # Correct MESS-PF single-decimal rounding error for consistency
    Ts = [298.15 if T == 298.2 else T for T in Ts]
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
