"""Test xarray_wrappers.py's functions."""

import numpy

from autoreact.ktp_xarray import xarray_wrappers

Temps = [1000, 1500, 2000, 2500]
Press = [1, 10, numpy.inf]
Rates = [[1e1, 1e2, 1e3, 1e4], [1e5, 1e6, 1e7, 1e8], [1e9, 1e10, 1e11, 1e12]]

Ktp = xarray_wrappers.from_data(Temps, Press, Rates)
Ktp_dct = {
    1.0: (([1000.0, 1500.0, 2000.0, 2500.0]), ([10.0, 100.0, 1000.0, 10000.0])),
    10: (([1000.0, 1500.0, 2000.0, 2500.0]), ([1.0e05, 1.0e06, 1.0e07, 1.0e08])),
    numpy.inf: (([1000.0, 1500.0, 2000.0, 2500.0]), ([1.0e09, 1.0e10, 1.0e11, 1.0e12])),
}
print(Ktp)


def test_get_temperatures():
    """Test the get_temperatures function."""
    temp = xarray_wrappers.get_temperatures(Ktp)
    print(temp)


def test_get_pressures():
    """Test the get_pressures function."""
    pres = xarray_wrappers.get_pressures(Ktp)
    print(f"pres = {pres}")
    print(type(pres))


def test_get_values():
    """Test the get_values function."""
    vals = xarray_wrappers.get_values(Ktp)
    print(f"vals = {vals}")
    print(type(vals))


def test_get_pslice():
    """Test the get_pslice function."""
    pslice = xarray_wrappers.get_pslice(Ktp, numpy.inf)
    print(f"pslice = {pslice}")
    print(type(pslice))


def test_get_tslice():
    """Test the get_tslice function."""
    tslice = xarray_wrappers.get_tslice(Ktp, 1500)
    print(tslice)


def test_get_spec_vals():
    """Test the get_spec_values function."""
    vals = xarray_wrappers.get_spec_vals(Ktp, 1500, 1)
    print(vals)


def test_get_ipslice():
    """Test the get_ipslice function."""
    ipslice = xarray_wrappers.get_ipslice(Ktp, 0)
    print(ipslice)


def test_get_itslice():
    """Test the get_itslice function."""
    itslice = xarray_wrappers.get_itslice(Ktp, 0)
    print(itslice)


def test_set_rates():
    """Test the set_rates function."""
    new_rates = xarray_wrappers.set_rates(Ktp, numpy.nan, 10, 2000)
    print(new_rates)


def test_dict_from_xarray():
    """Test the ktp_to_xarray function."""
    ktp_dct = xarray_wrappers.dict_from_xarray(Ktp)
    print(ktp_dct)


# def test_xarray_from_dict():
#    """Test the set_ktp_dct function."""
#    xarray = xarray_wrappers.xarray_from_dict(Ktp_dct)
#    print(xarray)


# test_get_pressures()
# test_get_temperatures()
# test_get_values()
test_get_pslice()
# test_get_tslice()
# test_get_spec_vals()
# test_get_ipslice()
# test_get_itslice()
# test_set_rates()
# test_dict_from_xarray()
# test_xarray_from_dict()
