"""Thermodynamic data."""

import pyparsing as pp

from ..util import FormulaData, chemkin, form
from ..util.type_ import Frozen
from . import data
from .data import Nasa7ThermFit, Therm_


class Species(Frozen):
    """A species with thermodynamic data."""

    name: str
    therm: Therm_


def from_chemkin_string(spc_str: str) -> Species:
    """Read species thermo from Chemkin string.

    :param spc_therm_str: Chemkin species therm string
    :return: Species thermo
    """
    # Parse string
    res = chemkin.parse_thermo(spc_str)

    # Extract thermo data
    therm_fit = data.from_chemkin_parse_results(res)

    return Species(name=res.name, therm=therm_fit)


def from_messpf_output_string(
    pf_str: str, formula: FormulaData, name: str | None = None, charge: int = 0
) -> data.Therm:
    """Build species thermo from MESS-PF output string.

    :param pf_str: MESS-PF output string
    :param formula: Species formula, as string or dict
    :param name: Species name
    :param charge: Molecular charge
    :return: Species thermo
    """
    if name is None:
        prefix_expr = pp.Literal("T,") + pp.Literal("K")
        name_expr = pp.Word(pp.printables)
        expr = pp.SkipTo(prefix_expr) + prefix_expr + name_expr("name")
        name = expr.parse_string(pf_str).get("name")

    therm = data.from_messpf_output_string(pf_str, formula=formula, charge=charge)
    return Species(name=name, therm=therm)


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
        form_dct=therm.formula,
        T_low=T_low,
        T_high=T_high,
        T_mid=T_mid,
        charge=therm.charge,
    )
    lines = chemkin.write_therm_entry_coefficient_lines(coeffs)

    lines = [line1, *lines]
    return "\n".join(f"{L: <78}{i + 1:>2d}" for i, L in enumerate(lines))


def pac99_input_string(spc: Species) -> str:
    """Generate a PAC99 input string for fitting to a NASA-7 polynomial.

    :param spc: Species thermo with thermo data (not a fit)
    :return: PAC99 input string
    """
    lines = [
        f"NAME  {spc.name}",
        f"{form.string(spc.therm.formula): <24}HF298",
    ]
    return "\n".join(lines)
