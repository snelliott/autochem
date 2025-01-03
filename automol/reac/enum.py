"""Functions for enumerating reactions."""

from collections.abc import Callable, Mapping, Sequence

from .. import amchi, graph, smiles
from .. import smarts as smarts_
from ._0core import Reaction, from_data


def from_smiles(
    smarts: str,
    rct_smis: Sequence[str] | Mapping[int, str],
    prd_smis: Sequence[str | None] | Mapping[int, str | None] | None = None,
) -> list[Reaction]:
    """Enumerate reactions from SMILES.

    Reagents can be given as lists or dictionaries by position in the SMARTS template.

    :param smarts: Reaction SMARTS string
    :param rct_chis: Reactant ChIs
    :param prd_chis: Product ChIs
    :return: List of Reaction objects
    """
    rct_gras, prd_gras = reagents(smarts, rct_smis, prd_smis, conv_=smiles.graph)
    return from_graphs(smarts, rct_gras, prd_gras)


def from_amchis(
    smarts: str,
    rct_chis: Sequence[str] | Mapping[int, str],
    prd_chis: Sequence[str | None] | Mapping[int, str | None] | None = None,
) -> list[Reaction]:
    """Enumerate reactions from AMChIs.

    Reagents can be given as lists or dictionaries by position in the SMARTS template.

    :param rct_chis: Reactant ChIs
    :param smarts: Reaction SMARTS string
    :param prd_chis: Product ChIs
    :return: List of Reaction objects
    """
    rct_gras, prd_gras = reagents(smarts, rct_chis, prd_chis, conv_=amchi.graph)
    return from_graphs(smarts, rct_gras, prd_gras)


def from_graphs(
    smarts: str,
    rct_gras: Sequence[object] | Mapping[int, object],
    prd_gras: Sequence[object | None] | Mapping[int, object | None] | None = None,
) -> list[Reaction]:
    """Enumerate reactions from graphs.

    Reagents can be given as lists or dictionaries by position in the SMARTS template.

    :param smarts: SMARTS string
    :param rct_gras: Reactant graphs
    :param prd_gras: Product graphs
    :return: List of Reaction objects
    """
    # Prepare reactants graph
    rct_gras, prd_gras = reagents(smarts, rct_gras, prd_gras, conv_=graph.explicit)
    rct_gras, _ = graph.standard_keys_for_sequence(rct_gras)
    rcts_gra = graph.union_from_sequence(rct_gras)

    # Complain if product graphs are given
    if any(g is not None for g in prd_gras):
        raise NotImplementedError("Product graphs not yet suppported")

    # Enumerate reactions
    ts_gras = graph.enum.reactions(smarts, rcts_gra)

    # Build reaction objects
    rcts_keys = list(map(sorted, map(graph.atom_keys, rct_gras)))
    rxns = []
    for ts_gra in ts_gras:
        prds_gra = graph.ts.products_graph(ts_gra, stereo=False, dummy=False)
        prd_gras_ = graph.connected_components(prds_gra)
        prds_keys = list(map(sorted, map(graph.atom_keys, prd_gras_)))
        rxns.append(from_data(ts_gra, rcts_keys, prds_keys))
    return rxns


# helpers
def reagents(
    smarts: str,
    rct_vals: Sequence[object] | Mapping[int, object],
    prd_vals: Sequence[object] | Mapping[int, object] | None = None,
    conv_: Callable[[object], object] | None = None,
) -> tuple[list[object], list[object]]:
    """Get lists of reagent values for a SMARTS reaction template.

    :param smarts: SMARTS string
    :param rct_vals: Reagent values
    :param prd_vals: Product values
    :return: Lists of reactant and product values
    """
    nrcts, nprds = smarts_.shape(smarts)
    rct_vals = dict(enumerate(rct_vals)) if isinstance(rct_vals, Sequence) else rct_vals
    prd_vals = (
        dict(enumerate(prd_vals))
        if isinstance(prd_vals, Sequence)
        else {} if prd_vals is None else prd_vals
    )

    assert isinstance(rct_vals, Mapping), f"rct_vals = {rct_vals}"
    assert isinstance(prd_vals, Mapping), f"prd_vals = {prd_vals}"

    rct_vals = list(map(rct_vals.get, range(nrcts)))
    prd_vals = list(map(prd_vals.get, range(nprds)))

    if conv_ is not None:
        rct_vals = [v if v is None else conv_(v) for v in rct_vals]
        prd_vals = [v if v is None else conv_(v) for v in prd_vals]

    return rct_vals, prd_vals
