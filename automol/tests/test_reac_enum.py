"""Test reac."""

import pytest
from automol import graph, reac


@pytest.mark.parametrize(
    "smarts, rct_smis, nrxns",
    [
        (graph.enum.ReactionSmarts.abstraction, ["C1=CCCC1", "[OH]"], 3),
    ],
)
def test__from_smiles(smarts, rct_smis, nrxns):
    """Test reac.from_smiles."""
    rxns = reac.enum.from_smiles(smarts, rct_smis)
    assert len(rxns) == nrxns, f"\nrxns = {rxns}"


if __name__ == "__main__":
    test__from_smiles(graph.enum.ReactionSmarts.abstraction, ["C1=CCCC1", "[OH]"], 3)
