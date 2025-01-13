"""Functions for enumerating reactions."""

from collections.abc import Sequence

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction

from ._00core import (
    explicit,
    relabel,
    ts_graph_from_reactants_and_products,
    union_from_sequence,
)
from ._02algo import connected_components, unique
from ._12rdkit import from_graph as to_rdkit
from ._12rdkit import to_graph as from_rdkit


# Reaction enumeration
class ReactionSmarts:
    """SMARTS reaction templates for enumeration."""

    pi2_addition = "[*:1]=[*:2].[OX1v1:3]>>[*:1]-[*:2]-[OX1v1:3]"
    elimination = "[H:5][C:1][C:2][O:3][OX1v1:4]>>[C:1]=[C:2].[OX1v1:3][O:4][H:5]"
    abstraction = "[C:1][H:2].[OX1v1:3]>>[C:1].[H:2][OX1v1:3]"


def reactions(smarts: str, gra: object, symeq: bool = False) -> list[object]:
    """Enumerate reaction TS graphs for a given SMARTS reaction template.

    :param smarts: SMARTS pattern for the reaction
    :param gra: Molecular graph representing the reactants
    :param symeq: Whether to include symmetrically equivalent reactions
    :return: TS graphs
    """
    ts_gras = [
        ts_graph_from_reactants_and_products(gra, p) for p in products(smarts, gra)
    ]
    return ts_gras if symeq else unique(ts_gras)


def products(smarts: str, gra: object) -> list[object]:
    """Enumerate products for a given SMARTS reaction template.

    :param smarts: SMARTS pattern for the reaction
    :param gra: Molecular graph representing the reactants
    :returns: Products graphs
    """
    assert gra == explicit(gra), f"Graph must be explicit\ngra = {gra}"

    # Form the reaction object
    rxn = AllChem.ReactionFromSmarts(smarts)

    # Form the reactant graphs
    rct_gras = connected_components(gra)
    # (label=True) stores the current graph keys to the `molAtomMapNumber` property
    rct_rdms = [to_rdkit(g, exp=True, label=True) for g in rct_gras]

    # Enumerate the products
    pos_dct = _template_map_number_to_reactant_position(rxn)
    prds_gras = [
        _products_graph(rct_rdms, prd_rdms, pos_dct)
        for prd_rdms in rxn.RunReactants(rct_rdms)
    ]
    return prds_gras


def _products_graph(
    rct_rdms: Sequence[Mol], prd_rdms: Sequence[Mol], pos_dct: dict[int, int]
) -> object:
    """Map atom keys to reactant positions.

    :param rct_rdms: RDKit reactant molecules
    :param prd_rdms: RDKit product molecules
    :param pos_dct: Mapping of atom map numbers to reactant positions
    :returns: Products graph
    """
    offset_dct = {p + 1: m.GetNumAtoms() for p, m in enumerate(rct_rdms)}

    prd_gras = []
    for prd_pos, prd_rdm in enumerate(prd_rdms):
        key_dct = {}
        for rda in prd_rdm.GetAtoms():
            prop_dct: dict[str, object] = rda.GetPropsAsDict()
            if "molAtomMapNumber" in prop_dct:
                key_dct[rda.GetIdx()] = prop_dct.get("molAtomMapNumber")
            else:
                map_num = rda.GetPropsAsDict().get("old_mapno")
                rct_pos = pos_dct.get(map_num, prd_pos)
                rct_key = rda.GetPropsAsDict().get("react_atom_idx")
                key_dct[rda.GetIdx()] = rct_key + offset_dct.get(rct_pos, 0)
        prd_gra = _product_graph_from_rdkit(prd_rdm)
        prd_gras.append(relabel(prd_gra, key_dct))

    return union_from_sequence(prd_gras)


def _template_map_number_to_reactant_position(rxn: ChemicalReaction) -> dict:
    """Map atom map numbers to reactant positions."""
    map2pos_dct = {}
    for pos in range(rxn.GetNumReactantTemplates()):
        rdt = rxn.GetReactantTemplate(pos)
        for rda in rdt.GetAtoms():
            map2pos_dct[rda.GetAtomMapNum()] = pos
    return map2pos_dct


def _product_graph_from_rdkit(rdm: Mol) -> object:
    """Sanitize an RDKit molecule with radicals."""
    for rda in rdm.GetAtoms():
        rda.SetNoImplicit(True)
    Chem.SanitizeMol(rdm)
    return from_rdkit(rdm)
