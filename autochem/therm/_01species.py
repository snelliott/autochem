"""Thermodynamic data."""

from ..util import chemkin
from ..util.type_ import Frozen
from ._00therm import Nasa7ThermFit, extract_thermo_from_chemkin_parse_results


class Species(Frozen):
    """A species with thermodynamic data."""

    name: str
    formula: dict[str, int]
    therm: object
    charge: int = 0


def from_chemkin_string(spc_str: str) -> Species:
    """Read species thermo from Chemkin string.

    :param spc_therm_str: Chemkin species therm string
    :return: Species thermo
    """
    # Parse string
    res = chemkin.parse_thermo(spc_str)

    # Extract thermo data
    therm_data = extract_thermo_from_chemkin_parse_results(res)

    # Determine charge, if any
    charge = 0
    form_dct = res.formula.copy()
    if "E" in form_dct:
        charge = form_dct.pop("E")

    return Species(
        name=res.name,
        formula=form_dct,
        charge=charge,
        therm=therm_data,
    )


def chemkin_string(spc: Species) -> str:
    """Get Chemkin thermo string.

    :param therm: Species thermo
    :return: Chemkin thermo string
    """
    therm = spc.therm
    T_low = T_high = T_mid = None
    match therm:
        case Nasa7ThermFit():
            T_low = therm.T_low
            T_high = therm.T_high
            T_mid = therm.T_mid
            coeffs = therm.coeffs_high + therm.coeffs_low
        case _:
            raise NotImplementedError(
                f"Thermodynamic data type {type(therm)} not implemented"
            )

    line1 = chemkin.write_therm_entry_header(
        name=spc.name,
        form_dct=spc.formula,
        T_low=T_low,
        T_high=T_high,
        T_mid=T_mid,
        charge=spc.charge,
    )
    lines = chemkin.write_therm_entry_coefficient_lines(coeffs)

    lines = [line1, *lines]
    return "\n".join(f"{L: <78}{i + 1:>2d}" for i, L in enumerate(lines))
