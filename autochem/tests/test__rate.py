"""Test autochem.rate."""

import numpy
import pytest

from autochem import rate

# Elementary
SIMPLE = {
    "units": {"energy": "kcal"},
    "chemkin": """
H(4)+HO2(10)=H2(2)+O2(3)                            2.945e+06 2.087     -1.455   
""",
}

# Three-body (now same type as Elementary in Cantera)
THREEBODY = {
    "units": {"energy": "cal"},
    "chemkin": """
H2(2)+M=H(4)+H(4)+M                                 4.580e+19 -1.400    104.40
H2(2)/2.55/ N2/1.01/ CO(1)/1.95/ CO2(12)/3.83/ H2O(7)/12.02/ 
""",
}

# Falloff
FALLOFF = {
    "units": {"energy": "kcal"},
    "chemkin": """
CO(1)+O(5)(+M)=CO2(12)(+M)                          1.880e+11 0.000     2.430    
CO2(12)/3.80/ CO(1)/1.90/ H2O(7)/12.00/ 
    LOW/ 1.400e+21 -2.100    5.500    /
""",
}

# Falloff (Troe)
FALLOFF_TROE = {
    "units": {"energy": "cal"},
    "chemkin": """
C2H4(+M)=H2+H2CC(+M)     8.000E+12     0.440    88770
    LOW  /               7.000E+50    -9.310    99860  /
    TROE /   7.345E-01   1.800E+02   1.035E+03   5.417E+03 /
    H2/2.000/   H2O/6.000/   CH4/2.000/   CO/1.500/  CO2/2.000/  C2H6/3.000/  AR/0.700/
""",
}

# Activated
ACTIVATED = {
    "units": {"energy": "cal"},
    "chemkin": """
CH3+CH3(+M)=H + C2H5(+M) 4.989E12 0.099 10600.0 ! Stewart
HIGH/ 3.80E-7 4.838 7710. / ! Chemically activated reaction
""",
}

# Activated (SRI)
ACTIVATED_SRI = {
    "units": {"energy": "cal"},
    "chemkin": """
CH3+CH3(+M)=H + C2H5(+M) 4.989E12 0.099 10600.0 ! Stewart
HIGH/ 3.80E-7 4.838 7710. / ! Chemically activated reaction
SRI / 1.641 4334 2725 / ! SRI pressure dependence
""",
}

# Plog
PLOG = {
    "units": {"energy": "cal"},
    "chemkin": """
C2H4+OH=PC2H4OH          2.560E+36    -7.752     6946   ! Arrhenius parameters at 1 atm
    PLOG /   1.000E-02   1.740E+43   -10.460     7699 /
    PLOG /   2.500E-02   3.250E+37    -8.629     5215 /
    PLOG /   1.000E-01   1.840E+35    -7.750     4909 /
    PLOG /   1.000E+00   2.560E+36    -7.752     6946 /
    PLOG /   1.000E+01   3.700E+33    -6.573     7606 /
    PLOG /   1.000E+02   1.120E+26    -4.101     5757 /
""",
}

# Cheb
CHEB = {
    "units": {"energy": "kcal"},
    "chemkin": """
CHO2(38)(+M)=H(4)+CO2(12)(+M)                       1.000e+00 0.000     0.000    
    TCHEB/ 300.000   2200.000 /
    PCHEB/ 0.010     98.702   /
    CHEB/ 6 4/
    CHEB/ 4.859e+00    9.247e-01    -3.655e-02   2.971e-15   /
    CHEB/ -3.682e-02   4.836e-02    2.294e-02    4.705e-17   /
    CHEB/ -3.061e-15   -4.107e-16   1.623e-17    -1.319e-30  /
    CHEB/ 3.682e-02    -4.836e-02   -2.294e-02   -4.705e-17  /
    CHEB/ -4.859e+00   -9.247e-01   3.655e-02    -2.971e-15  /
    CHEB/ 3.682e-02    -4.836e-02   -2.294e-02   -4.705e-17  /
""",
}


@pytest.mark.parametrize(
    "name, data, check_roundtrip",
    [
        ("SIMPLE", SIMPLE, True),
        ("THREEBODY", THREEBODY, True),
        ("FALLOFF", FALLOFF, True),
        ("FALLOFF_TROE", FALLOFF_TROE, True),
        ("ACTIVATED", ACTIVATED, True),
        ("ACTIVATED_SRI", ACTIVATED_SRI, True),
        ("PLOG", PLOG, True),
        ("CHEB", CHEB, False),
    ],
)
def test__from_chemkin_string(name, data, check_roundtrip: bool):
    units = data.get("units")
    chem_str = data.get("chemkin")

    # Read
    k = rate.from_chemkin_string(chem_str, units=units)

    # Evaluate
    T0 = 500
    T1 = [500, 600, 700, 800]
    P0 = 1.0
    P1 = [0.1, 1.0, 10.0]
    kT0P0 = k(T0, P0)
    assert numpy.shape(kT0P0) == (), kT0P0
    kT1P0 = k(T1, P0)
    assert numpy.shape(kT1P0) == (4,), kT1P0
    kT0P1 = k(T0, P1)
    assert numpy.shape(kT0P1) == (3,), kT0P1
    kT1P1 = k(T1, P1)
    assert numpy.shape(kT1P1) == (4, 3), kT1P1

    # Write
    chem_str_ = rate.chemkin_string(k)
    print(chem_str_)
    k_ = rate.from_chemkin_string(chem_str_)

    # Check roundtrip
    if check_roundtrip:
        assert k == k_, f"\n   {k}\n!= {k_}"

    # Plot against another rate
    copm_units = SIMPLE.get("units")
    copm_chem_str = SIMPLE.get("chemkin")
    comp_k = rate.from_chemkin_string(copm_chem_str, units=copm_units)
    rate.display(k, comp_rates=[comp_k], comp_labels=["comp"])


if __name__ == "__main__":
    # test__from_chemkin_string("FALLOFF", FALLOFF)
    # test__from_chemkin_string("THREEBODY", THREEBODY)
    # test__from_chemkin_string("PLOG", PLOG)
    test__from_chemkin_string("CHEB", CHEB, False)
