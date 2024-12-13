"""
Wrappers for the new xarray system. Constructors, Getters, then Setters.
"""

import xarray
import numpy

# Constructors
def from_data(temps, press, rates):
    """Construct a KTP DataArray from data"""

    ktp = xarray.DataArray(rates, (("pres", press), ("temp", temps)))

    return ktp



# Getters
def get_pressures(ktp):
    """Gets the pressure values"""

    return ktp.pres.data


def get_temperatures(ktp):
    """Gets the temperature values"""

    return ktp.temp.data


def get_values(ktp):
    """Gets the KTP values"""

    return ktp.values


def get_pslice(ktp, ip):
    """Get a slice at a selected pressure value"""

    return ktp.sel(pres=ip)


def get_tslice(ktp, it):
    """Get a slice at a selected temperature value"""

    return ktp.sel(temp=it)


def get_spec_vals(ktp, it, ip):
    """Get a specific value at a selected temperature and pressure value"""

    return ktp.sel(temp=it, pres=ip)


def get_ipslice(ktp, ip):
    """Get a slice at a selected pressure index"""

    return ktp.isel(pres=ip)


def get_itslice(ktp, it):
    """Get a slice at a selected temperature index"""

    return ktp.isel(temp=it)



# Setters
def set_rates(ktp, rates, pres, temp):
    """Sets the KTP values"""

    ktp.loc[{"pres": pres, "temp": temp}] = rates
    return ktp


# Translators

def dict_from_xarray(xarray_in):
    """Turns an xarray into a ktp_dct"""

    ktp_dct = {}
    #dict_temps = get_temperatures(xarray)
    for pres in get_pressures(xarray_in):
        dict_temps = get_temperatures(xarray_in)
        dict_kts = []
        curr_temps = numpy.copy(dict_temps)
        for temp_idx, temp in enumerate(dict_temps):
            kt = get_spec_vals(xarray_in, temp, pres)
            if numpy.isnan(kt):
                curr_temps = numpy.delete(curr_temps, temp_idx)
            else:
                dict_kts += (float(kt),)
        dict_temps = curr_temps
        dict_kts = numpy.array(dict_kts, dtype=numpy.float64)
        if pres == numpy.inf:
            pres = 'high'
        ktp_dct[pres] = (dict_temps, dict_kts)
    return ktp_dct

#Stopped working on this one because it was less critical!

#def xarray_from_dict(ktp_dct):
#    """DOES NOT WORK YET!
#    Turns a ktp_dct into an xarray"""
#
#    xarray_press = []
#    xarray_temps = []
#    for pressures, (temps,x) in ktp_dct.items():
#        xarray_press.append(pressures)
#        for temp in temps:
#            if temp not in xarray_temps:
#                xarray_temps.append(temp)
#    xarray_temps = xarray_temps.sort()
#    temporary_kts = numpy.ndarray((len(xarray_press),len(xarray_temps)))
#    for pres_idx, pres in enumerate(xarray_press):
#        ktp = list(ktp_dct[pres])[1]
#        for kt in ktp:
#            temporary_kts[pres_idx] = kt
#            breakpoint()
#    xarray = from_data(xarray_temps, xarray_press, temp_kts)
#    breakpoint()
#    return xarray
