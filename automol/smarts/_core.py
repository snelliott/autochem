"""SMARTS functions."""

from . import rd


# properties
def shape(smarts: str) -> tuple[list[int], list[int]]:
    """Get numbers of the reactants and products in SMARTS string.

    :param smarts: SMARTS string
    :return: Reaction shape
    """
    return rd.shape(rd.from_smarts(smarts))
