"""Thermodynamic data."""

import abc
from typing import ClassVar

import numpy

from ..util import chemkin
from ..util.type_ import Frozen, Scalable, Scalers, SubclassTyped


class BaseThermoData(Frozen, Scalable, SubclassTyped, abc.ABC):
    """Abstract base class for thermodynamic data."""

    T_low: float
    T_high: float


class ThermoData(BaseThermoData):
    """Raw thermodynamic data."""

    pass


class ThermoDataFit(BaseThermoData, abc.ABC):
    """Fitted thermodynamic data."""

    pass


class Nasa7ThermoDataFit(ThermoDataFit):
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


def extract_thermo_from_chemkin_parse_results(
    res: chemkin.ChemkinThermoParseResults,
    T_mid: float = 1000,  # noqa: N803
) -> ThermoDataFit:
    """Extract thermo data from Chemkin parse results.

    :param res: Chemkin thermo parse results
    :param T_mid: Default mid-point temperature
    :return: Thermo data
    """
    if len(res.coeffs) == 14:
        return Nasa7ThermoDataFit(
            T_low=res.T_low,
            T_high=res.T_high,
            coeffs_low=res.coeffs[7:],
            coeffs_high=res.coeffs[:7],
            T_mid=res.T_mid or T_mid,
        )

    raise ValueError(f"Unable to interpret parse results: {res}")
