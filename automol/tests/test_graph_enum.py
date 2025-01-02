"""test graph.enum."""

import pytest
from automol import graph, smiles


@pytest.mark.parametrize(
    "rcts_smi, smarts, nrxns",
    [("CCC.[OH]", graph.enum.ReactionTemplate.abstraction, 2)],
)
def test__reactions(rcts_smi, smarts, nrxns):
    """Test reactions."""
    rcts_gra = smiles.graph(rcts_smi)
    ts_gras = graph.enum.reactions(rcts_gra, smarts)
    assert len(ts_gras) == nrxns, f"ts_gras = {ts_gras}"


if __name__ == "__main__":
    test__reactions("CCC.[OH]", graph.enum.ReactionTemplate.abstraction, 2)
