"""Formula utilities."""

from typing import TypeAlias

import pyparsing as pp
from pyparsing import pyparsing_common as ppc

SYMBOL = pp.Word(pp.alphas.upper(), pp.alphas.lower())
COUNT = pp.Opt(ppc.integer).setParseAction(lambda x: x if x else [1])
STOICH = pp.Group(SYMBOL + COUNT)
FORMULA = pp.OneOrMore(STOICH)
Formula: TypeAlias = dict[str, int]
FormulaData: TypeAlias = Formula | str


def normalize_input(fml_inp: FormulaData) -> Formula:
    """Normalize formula input, which could be given as a string.

    :param fml_inp: Formula input
    :return: Formula
    """
    fml = (
        dict(FORMULA.parse_string(fml_inp).as_list())
        if isinstance(fml_inp, str)
        else fml_inp
    )
    fml = {str.title(k): int(v) for k, v in fml.items() if v}
    return fml


def string(fml_inp: FormulaData, upper: bool=False) -> str:
    """Convert formula input to string.

    :param fml_inp: Formula input
    :param upper: Whether to write all-uppercase symbols
    :return: Formula string
    """
    fml = normalize_input(fml_inp)
    fml = {k.upper(): v for k, v in fml.items()} if upper else fml
    return "".join([f"{k}{v}" if v > 1 else k for k, v in fml.items()])
