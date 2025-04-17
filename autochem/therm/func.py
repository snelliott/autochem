"""Single-temperature-range thermodynamic fit function models."""

import abc
from typing import ClassVar, Literal

import numpy
import pydantic
from numpy.typing import ArrayLike, NDArray

from .. import unit_
from ..unit_ import UNITS, C, D, UnitsData
from ..util.type_ import Frozen, Scalable, SubclassTyped


class Bounded(pydantic.BaseModel):
    T_min: float
    T_max: float

    def in_bounds(
        self,
        T: ArrayLike,  # noqa: N803
        include_max: bool = True,
    ) -> NDArray[numpy.bool_]:
        """Determine whether temperature(s) are in bounds.

        :param T: Temperature(s)
        :return: Boolean value(s)
        """
        T = numpy.array(T, dtype=numpy.float64)
        T_greater_than_min = T >= self.T_min
        T_less_than_max = (T <= self.T_max) if include_max else (T < self.T_max)
        return T_greater_than_min & T_less_than_max

    def all_in_bounds(
        self,
        T: ArrayLike,  # noqa: N803
    ) -> bool:
        """Determine whether all temperature(s) are in bounds.

        :param T: Temperature(s)
        :return: `True` if they are
        """
        return numpy.all(self.in_bounds(T))

    def assert_all_in_bounds(
        self,
        T: ArrayLike,  # noqa: N803
    ) -> None:
        """Assert that all temperature(s) are in bounds.

        :param T: Temperature(s)
        """
        assert self.all_in_bounds(T), f"{self.T_min} !<= {T} !<= {self.T_max}"


class ThermCalculator(Frozen, abc.ABC):
    """Abstract base class for fit functions."""

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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


class Nasa7Calculator(Bounded, ThermCalculator):
    a0: float
    a1: float
    a2: float
    a3: float
    a4: float
    a5: float
    a6: float

    # Private attributes
    type_: ClassVar[str] = "nasa7"

    @unit_.manage_units([], D.energy_per_substance / D.temperature)
    def heat_capacity(
        self,
        T: ArrayLike,  # noqa: N803
        const: Literal["P", "V"] = "P",
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate heat capacity, C_V(T) or C_P(T).

        Formula:
            C_P(T) = R (a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4)
            C_V(T) = C_P(T) - R

        :param T: Temperature(s)
        :param const: Whether to hold pressure ("P") or volume ("V") constant
        :param units: Unit system
        :return: Function value(s)
        """
        self.assert_all_in_bounds(T)

        R = unit_.const.value(C.gas, UNITS)
        T = numpy.array(T, dtype=numpy.float64)
        C_ = R * (
            self.a0 + self.a1 * T + self.a2 * T**2 + self.a3 * T**3 + self.a4 * T**4
        )
        C_ -= R if const == "V" else 0.0
        return C_

    @unit_.manage_units([], D.energy_per_substance)
    def enthalpy(
        self,
        T: ArrayLike,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate enthalpy, H(T).

        Formula:
            H(T) = R (a0 T + (a1/2) T^2 + (a2/3) T^3 + (a3/4) T^4 + (a4/5) T^5 + a5)

        Coefficient a5 is defined to satisfy H(298.15) = heat of formation at 298.15.

        :param T: Temperature(s)
        :param units: Unit system
        :return: Function value(s)
        """
        self.assert_all_in_bounds(T)

        R = unit_.const.value(C.gas, UNITS)
        T = numpy.array(T, dtype=numpy.float64)
        H = R * (
            self.a0 * T
            + (self.a1 / 2) * T**2
            + (self.a2 / 3) * T**3
            + (self.a3 / 4) * T**4
            + (self.a4 / 5) * T**5
            + self.a5
        )
        return H

    @unit_.manage_units([], D.energy_per_substance / D.temperature)
    def entropy(
        self,
        T: ArrayLike,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float64]:
        """Evaluate entropy, S(T).

        Formula:
            S(T) = R (a0 ln(T) + a1 T + (a2/2) T^2 + (a3/3) T^3 + (a4/4) T^4 + a6)

        :param T: Temperature(s)
        :param units: Unit system
        :return: Function value(s)
        """
        self.assert_all_in_bounds(T)

        R = unit_.const.value(C.gas, UNITS)
        T = numpy.array(T, dtype=numpy.float64)
        S = R * (
            self.a0 * numpy.log(T)
            + self.a1 * T
            + (self.a2 / 2) * T**2
            + (self.a3 / 3) * T**3
            + (self.a4 / 4) * T**4
            + self.a6
        )
        return S
