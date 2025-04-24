"""Test autochem.rate."""

import pytest

from autochem import therm

NASA7 = """
! NASA polynomial coefficients for CH2O3
CH2O3(82)               H   2C   1O   3     G   100.000  5000.000  956.80      1
    1.18953474E+01 1.71602859E-03-1.47932622E-07 9.25919346E-11-1.38613705E-14    2
-7.81027323E+04-3.82545468E+01 2.98116506E+00 1.14388942E-02 2.77945768E-05    3
-4.94698411E-08 2.07998858E-11-7.51362668E+04 1.09457245E+01                   4
"""


@pytest.mark.parametrize(
    "name, spc_str0",
    [
        ("NASA7", NASA7),
    ],
)
def test__from_chemkin_string(name, spc_str0):
    # Dictionary serialization/deserialization
    spc = therm.from_chemkin_string(spc_str0)
    spc_ = therm.Species.model_validate(spc.model_dump())
    assert spc == spc_, f"\n   {spc}\n!= {spc_}"

    # Chemkin serialization/deserialization
    spc_str = therm.chemkin_string(spc)
    spc_ = therm.from_chemkin_string(spc_str)
    assert spc == spc_, f"\n   {spc}\n!= {spc_}"

    # Scale
    spc_times_2 = spc * 2
    spc_divided_by_2 = spc / 2

    # Plot
    therm.display(spc)
    therm.display(
        spc,
        label="original",
        others=[spc_times_2, spc_divided_by_2],
        others_labels=["doubled", "halved"],
    )


if __name__ == "__main__":
    test__from_chemkin_string("NASA7", NASA7)