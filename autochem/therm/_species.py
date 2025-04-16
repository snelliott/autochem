"""Thermodynamic data."""

import datetime

import pyparsing as pp

from ..unit_ import UnitsData
from ..util import FormulaData, chemkin, form
from ..util.type_ import Frozen
from . import data
from .data import Nasa7ThermFit, Therm, Therm_


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
    pf_str: str,
    formula: FormulaData,
    name: str | None = None,
    charge: int = 0,
    Hf: float | None = None,  # noqa: N803
    Tf: float = 0,  # noqa: N803
    units: UnitsData | None = None,
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

    therm = data.from_messpf_output_string(
        pf_str, formula=formula, charge=charge, Hf=Hf, Tf=Tf, units=units
    )
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


def pac99_input_string(
    spc: Species,
    Tmin: float = 200,  # noqa: N803
    Tmid: float = 1000,  # noqa: N803
    Tmax: float = 3000,  # noqa: N803
) -> str:
    """Generate a PAC99 input string for fitting to a NASA-7 polynomial.

    :param spc: Species thermo with thermo data (not a fit)
    :return: PAC99 input string
    """
    assert isinstance(spc.therm, Therm)
    fml_str = form.string(spc.therm.formula)
    Hf298 = spc.therm.enthalpy_of_formation_room_temperature(units={"energy": "J"})
    Ts = spc.therm.Ts
    # Enthalpy units are set to kJ by "KJOULE" keyword below
    dH298 = spc.therm.delta_enthalpy(T=298, method="nearest", units={"energy": "kJ"})
    dHs = spc.therm.delta_enthalpy_data(units={"energy": "kJ"})
    # Entropy and heat capacity are always in Joules
    # (see https://ntrs.nasa.gov/citations/19930003779, p. 33)
    Ss = spc.therm.entropy_data(P=1, units={"pressure": "bar", "energy": "J"})
    Cs = spc.therm.heat_capacity_data(at_const_P=True, units={"energy": "J"})
    date = format(datetime.date.today(), r"%Y%m%d")
    return "\n".join(
        [
            pac99_input_line("NAME", spc.name),
            pac99_input_line(fml_str, None, None, "HF298", Hf298, "JOULES", decimals=0),
            pac99_input_line("DATE", date),
            pac99_input_line("REFN", "ME"),
            pac99_input_line(
                "LSTS", "OLD", None, "T", Tmin, "T", Tmin, "T", Tmid, decimals=0
            ),
            pac99_input_line("LSTS", "T", Tmax, decimals=0),
            pac99_input_line("OUTP", "LSQS"),
            pac99_input_line("METH", "READIN", None, "KJOULE", None, "BAR"),
            pac99_input_line(None, "T", 0, "CP", 0, "S", 0, "H-H2", -dH298),
            *(
                pac99_input_line(None, "T", T, "CP", C, "S", S, "H-H0", dH)
                for T, C, S, dH in zip(Ts, Cs, Ss, dHs, strict=True)
            ),
            pac99_input_line("FINISH"),
        ]
    )


def pac99_input_line(
    key: str | None = None,
    label1: str | None = None,
    num1: float | None = None,
    label2: str | None = None,
    num2: float | None = None,
    label3: str | None = None,
    num3: float | None = None,
    label4: str | None = None,
    num4: float | None = None,
    decimals: int = 3,
) -> str:
    """Format a line of PAC99 input."""
    widths = [6] + [6, 12] * 4
    vals = [
        key,
        label1,
        pac99_number(num1, decimals=decimals),
        label2,
        pac99_number(num2, decimals=decimals),
        label3,
        pac99_number(num3, decimals=decimals),
        label4,
        pac99_number(num4, decimals=decimals),
    ]
    line = ""
    for idx, (width, val) in enumerate(zip(widths, vals, strict=True)):
        if val is not None:
            indent = sum(widths[:idx])
            line = f"{line: <{indent}}{val: <{width}}"
    return line


def pac99_number(num: float | None = None, decimals: int = 3) -> str:
    """Format number for PAC99 input."""
    if num is None:
        return None

    num_str = f"{num:.{decimals}f}"
    num_str = num_str if "." in num_str else f"{num_str}."
    return f"{num_str: >10}  "
