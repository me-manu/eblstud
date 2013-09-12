"""
Available Functions are:
DecDMS2float(dec)
RaHMS2float(ra)
angsep(r1,r2,d1,d2)angsep(r1,r2,d1,d2)
"""

def DecDMS2float(dec):
    """
    Convert string with declination to float in degrees
    
    Parameters
    ----------
    dec:	string, declination in format +/-00d00m00.0s 

    Returns
    -------
    declination as float in degrees
    
    """
    dec = dec.lower()
    result = float(dec[0:dec.find("d")]) 
    if result < 0.:
	sign = -1.
    else:
	sign = 1.
    result += sign*float(dec[dec.find("d")+1:dec.find("m")])/60. 
    result += sign*float(dec[dec.find("m")+1:dec.find("s")])/3600.
    if result > 90. or result < -90.:
	raise ValueError("Declination is out of bounds [-90deg : +90deg]")
    return (result)

def RaHMS2float(ra):
    """ 
    Convert string with Right Ascension to float in degrees
    
    Parameters
    ----------
    ra:	string, ra in format at 00h00m00.0s 

    Returns
    -------
    declination as float in degrees
    """
    ra = ra.lower()
    result = float(ra[0:ra.find("h")]) % 24.
    result += float(ra[ra.find("h")+1:ra.find("m")])/60. 
    result += float(ra[ra.find("m")+1:ra.find("s")])/3600.
    result *= 15.
    return (result)

import numpy as np
from numpy import pi
# Angular seperation in degrees
def angsep(r1,r2,d1,d2):
    """Calculate angulate seperation in degrees for two (RA,DEC) tuples in degrees

    Parameters:
    -----------
    r1: float or n-dim array, RA of 1st source
    r2: float or n-dim array, DEC of 1st source
    d1: float or n-dim array, RA of 2nd source
    d2: float or n-dim array, DEC of 2nd source

    Returns
    -------
    angular seperation in degrees
    """

    if np.isscalar(r1):
	r1 = np.array([r1])
    if np.isscalar(r2):
	r2 = np.array([r2])
    if np.isscalar(d1):
	d1 = np.array([d1])
    if np.isscalar(d2):
	d2 = np.array([d2])
    th1 = (90.0*np.ones(d1.shape)-d1)*pi/180.
    th2 = (90.0*np.ones(d2.shape)-d2)*pi/180.  
    ph1 = r1 * pi/180.
    ph2 = r2 * pi/180.
    cth = np.sin(th1)*np.sin(th2)*\
	(np.cos(ph1)*np.cos(ph2)+np.sin(ph1)*np.sin(ph2))+np.cos(th1)*np.cos(th2)
    cth[np.where(cth > 1.)[0]] = np.zeros(cth[np.where(cth > 1.)[0]].shape)
    return np.arccos(cth)*180./pi
