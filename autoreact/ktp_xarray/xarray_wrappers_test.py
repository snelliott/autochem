"""Tests xarray_wrappers.py's functions"""

import numpy
from autoreact.ktp_xarray import xarray_wrappers

Temps = [1000, 1500, 2000, 2500]
Press = [1, 10, numpy.inf]
Rates = [[1e1, 1e2, 1e3, 1e4], [1e5, 1e6, 1e7, 1e8], [1e9, 1e10, 1e11, 1e12]]

Ktp = xarray_wrappers.from_data(Temps, Press, Rates)
Ktp_dct = {1.0: (([1000., 1500., 2000., 2500.]),
                ([ 10., 100., 1000., 10000.])),
           10: (([1000., 1500., 2000., 2500.]),
                ([1.e+05, 1.e+06, 1.e+07, 1.e+08])),
           numpy.inf: (([1000., 1500., 2000., 2500.]),
                ([1.e+09, 1.e+10, 1.e+11, 1.e+12]))}
print(Ktp)

def test_get_temperatures():
    """Tests the get_temperatures function"""
    temp = xarray_wrappers.get_temperatures(Ktp)
    print(temp)


def test_get_pressures():
    """Tests the get_pressures function"""
    pres = xarray_wrappers.get_pressures(Ktp)
    print(pres)


def test_get_values():
    """Tests the get_values function"""
    vals = xarray_wrappers.get_values(Ktp)
    print(vals)


def test_get_pslice():
    """Tests the get_pslice function"""
    pslice = xarray_wrappers.get_pslice(Ktp, numpy.inf)
    print(pslice)


def test_get_tslice():
    """Tests the get_tslice function"""
    tslice = xarray_wrappers.get_tslice(Ktp, 1500)
    print(tslice)


def test_get_spec_vals():
    """Tests the get_spec_values function"""
    vals = xarray_wrappers.get_spec_vals(Ktp, 1500, 1)
    print(vals)


def test_get_ipslice():
    """Tests the get_ipslice function"""
    ipslice = xarray_wrappers.get_ipslice(Ktp, 0)
    print(ipslice)


def test_get_itslice():
    """Tests the get_itslice function"""
    itslice = xarray_wrappers.get_itslice(Ktp, 0)
    print(itslice)


def test_set_rates():
    """Tests the set_rates function"""
    new_rates = xarray_wrappers.set_rates(Ktp, numpy.nan, 10, 2000)
    print(new_rates)

def test_dict_from_xarray():
    """Tests the ktp_to_xarray function"""
    ktp_dct = xarray_wrappers.dict_from_xarray(Ktp)
    print(ktp_dct)

#def test_xarray_from_dict():
#    """Tests the set_ktp_dct function"""
#    xarray = xarray_wrappers.xarray_from_dict(Ktp_dct)
#    print(xarray)


test_get_pressures()
test_get_temperatures()
test_get_values()
test_get_pslice()
test_get_tslice()
test_get_spec_vals()
test_get_ipslice()
test_get_itslice()
test_set_rates()
test_dict_from_xarray()
#test_xarray_from_dict()
