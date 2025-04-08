"""Thermodynamic data."""

from ..util import chemkin
from ..util.type_ import Frozen
from ._00thermo import Nasa7ThermoDataFit, extract_thermo_from_chemkin_parse_results


class SpeciesThermo(Frozen):
    """Thermodynamic data for a species."""

    name: str
    formula: dict[str, int]
    therm_data: object
    charge: int = 0


def from_chemkin_string(therm_str: str) -> SpeciesThermo:
    """Read species thermo from Chemkin string.

    :param therm_str: Chemkin string
    :return: Species thermo
    """
    # Parse string
    res = chemkin.parse_thermo(therm_str)

    # Extract thermo data
    therm_data = extract_thermo_from_chemkin_parse_results(res)

    # Determine charge, if any
    charge = 0
    form_dct = res.formula.copy()
    if "E" in form_dct:
        charge = form_dct.pop("E")

    return SpeciesThermo(
        name=res.name,
        formula=form_dct,
        charge=charge,
        therm_data=therm_data,
    )


def chemkin_string(therm: SpeciesThermo) -> str:
    """Get Chemkin thermo string.

    :param therm: Species thermo
    :return: Chemkin thermo string
    """
    therm_data = therm.therm_data
    T_mid = None
    match therm_data:
        case Nasa7ThermoDataFit():
            T_mid = therm_data.T_mid
            coeffs = therm_data.coeffs_high + therm_data.coeffs_low
        case _:
            raise NotImplementedError(
                f"Thermodynamic data type {type(therm_data)} not implemented"
            )

    line1 = chemkin.write_therm_entry_header(
        name=therm.name,
        form_dct=therm.formula,
        T_low=therm.therm_data.T_low,
        T_high=therm.therm_data.T_high,
        T_mid=T_mid,
        charge=therm.charge,
    )
    lines = chemkin.write_therm_entry_coefficient_lines(coeffs)

    lines = [line1, *lines]
    return "\n".join(f"{L: <78}{i + 1:>2d}" for i, L in enumerate(lines))
