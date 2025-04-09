"""Unit system."""

import enum
import functools
import itertools
import operator
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, TypeAlias

import more_itertools as mit
import numpy
import pint
import pydantic
from numpy.typing import NDArray

from ..util.type_ import Frozen, Unit_

UR = pint.UnitRegistry()


# Model for specifying units
class Units(Frozen):
    """Unit system."""

    # Core units
    time: Unit_ = pint.Unit("s")
    temperature: Unit_ = pint.Unit("K")
    length: Unit_ = pint.Unit("cm")
    substance: Unit_ = pint.Unit("mol")
    pressure: Unit_ = pint.Unit("atm")
    energy: Unit_ = pint.Unit("cal")

    # Derived units
    @functools.cached_property
    def energy_per_substance(self) -> pint.Unit:
        """Energy per substance unit."""
        return pint.Unit(self.energy / self.substance)

    @functools.cached_property
    def concentration(self) -> pint.Unit:
        """Concentration unit."""
        return pint.Unit(self.substance / self.length**3)

    def rate_constant(self, order: int) -> pint.Unit:
        """Rate constant unit.

        :param order: Reaction order
        """
        if order == 1:
            return pint.Unit(self.time**-1)

        return pint.Unit(self.concentration ** (1 - order) * self.time**-1)


# Alias for unit-convertible data
UnitData: TypeAlias = str | pint.Unit
UnitsData: TypeAlias = Mapping[str, UnitData] | Units


# Internal unit system (defaults of above Units class)
UNITS = Units()


class Dimension:
    """Dimension class."""

    _data: dict[str, int]

    def __init__(
        self, arg: "str | Mapping[str, int] | Dimension | None" = None, **kwargs
    ):
        if arg is not None:
            assert not kwargs, f"Invalid arguments: {arg}, {kwargs}"
        if isinstance(arg, self.__class__):
            self._data = arg._data
        elif isinstance(arg, str):
            self._data = {arg: 1}
        elif isinstance(arg, Mapping):
            self._data = dict(arg)
        else:
            self._data = kwargs

        assert all(hasattr(UNITS, k) for k in self._data), (
            f"Invalid names: {self._data}"
        )
        self._data = {k: int(v) for k, v in self._data.items()}

    def items(self) -> list[tuple[str, int]]:
        """Get items of dimension."""
        return sorted(self._data.items(), key=lambda t: t[::-1], reverse=True)

    def __repr__(self) -> str:
        """Represent as string."""
        data_str = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({data_str})"

    def __pow__(self, power: int) -> "Dimension":
        """Raise dimension to a power.

        :param power: Power
        :return: New dimension
        """
        data = {k: v * power for k, v in self._data.items()}
        return self.__class__(data)

    def __mul__(self, other: "Dimension") -> "Dimension":
        """Multiply two dimensions together.

        :param other: Other dimension
        :return: New dimension
        """
        data = {
            k: self._data.get(k, 0) + other._data.get(k, 0)
            for k in set(self._data) | set(other._data)
        }
        return self.__class__(data)

    __rmul__ = __mul__

    def __truediv__(self, other: "Dimension") -> "Dimension":
        """Divide one dimension by another.

        :param other: Other dimension
        :return: New dimension
        """
        data = {
            k: self._data.get(k, 0) - other._data.get(k, 0)
            for k in set(self._data) | set(other._data)
        }
        return self.__class__(data)


class Dim:
    """Easy access to common dimensions."""

    time = Dimension("time")
    temperature = Dimension("temperature")
    length = Dimension("length")
    substance = Dimension("substance")
    pressure = Dimension("pressure")
    energy = Dimension("energy")
    energy_per_substance = Dimension("energy_per_substance")
    concentration = Dimension("concentration")
    rate_constant = Dimension("rate_constant")


# Alias for dimension-convertible data
DimensionData: TypeAlias = str | Mapping[str, int] | Dimension


# Constructors
def from_unit_sequence(unit_seq: Sequence[pint.Unit]) -> Units:
    """Construct unit sytem from sequence of composite units.

    :param unit_seq: Units sequence
    :param strict: Whether to raise an error if units are inconsistent
    :return: Unit system
    """
    # Split into components and chain them
    unit_iter = itertools.chain.from_iterable(
        map(pint.util.to_units_container, unit_seq)
    )
    unit_pool = list(map(pint.Unit, unit_iter))

    # For each dimension, gather matching components
    data = {}
    # for dim in UNITS.model_dump():
    for dim, unit0 in UNITS:
        # Separate out compatible units from the pool
        unit_pool, units = map(
            list, mit.partition(pint.Unit(unit0).is_compatible_with, unit_pool)
        )
        if units:
            # Pop the first one, make sure they match, and add it
            unit = pint.Unit(units.pop())
            assert all(map(unit.is_compatible_with, units)), f"{unit} !~ {units}"
            data[dim] = unit

    return Units.model_validate(data)


# Properties
def gas_constant_value(units: Units) -> float:
    """Determine gas constant value in unit system.

    :param units: Unit system
    :return: Gas constant value
    """
    return pint.Quantity("molar_gas_constant").m_as(
        units.energy_per_substance / units.temperature
    )


def dimension_unit(units: Units, dim: DimensionData, **kwargs) -> pint.Unit:
    """Determine the unit for a dimension.

    :param units: Unit system
    :param dim: Dimension
    :param **kwargs: Extra arguments for unit determination
    :return: Unit
    """
    dim = Dimension(dim)

    def _unit(dim_name: str) -> pint.Unit:
        if dim_name == "rate_constant":
            order = kwargs.get("order")
            assert order is not None, f"Missing 'order': {kwargs}"
            return units.rate_constant(order)
        return getattr(units, dim_name)

    return functools.reduce(operator.mul, (_unit(k) ** v for k, v in dim.items()))


def dimension_conversion_factor(
    units: Units, new_units: Units, dim: DimensionData, **kwargs
) -> float:
    """Convert dimension value to new units.

    :param units: Units sytem
    :param new_units: New units system
    :param dim: Dimension
    :param **kwargs: Extra arguments for unit determination
    :return: New value
    """
    unit = dimension_unit(units, dim, **kwargs)
    new_unit = dimension_unit(new_units, dim, **kwargs)
    return pint.Quantity(1, unit).m_as(new_unit)


def convert_dimension_value(
    units: Units, new_units: Units, dim: DimensionData, val: Any, **kwargs
) -> NDArray[numpy.float64]:
    """Convert dimension value to new units.

    :param units: Units sytem
    :param new_units: New units system
    :param dim: Dimension
    :param val: Value
    :param **kwargs: Extra arguments for unit determination
    :return: New value
    """
    unit = dimension_unit(units, dim, **kwargs)
    new_unit = dimension_unit(new_units, dim, **kwargs)
    new_val = pint.Quantity(val, unit).m_as(new_unit)
    return numpy.array(new_val, dtype=numpy.float64)
