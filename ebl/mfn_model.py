"""
Class to read Mean free path of EBL models generated with CreateMeanFreePathTable.py

History of changes:
Version 0.1
- Created 18th November 2011 - Happy Birthday!

"""

__version__ = 0.1
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

# ================================================================================#
# Imports
import numpy as np
import math
import scipy.integrate
from scipy.interpolate import RectBivariateSpline as BiSpline
from eblstud.misc.constants import *
import pickle
import gzip

# ================================================================================#

### WORKS AT THE MOMENT ONLY ABOVE 500 GEV!!! AND BELOW 30 TEV FOR Z=0.5!!!

# -- MFN Model from 2D Matrix ----------------------------------------------------------------#
class MFNModel:
    def __init__(self,file_name='None',model='kneiske'):
	"""
	Read MFN from 2D Matrix

	Matrix Entries Mean free path [m] (E,z) 
	First Index: Energy (rows), TeV
	Second Index: Redshift
	self.mfn is 2D Matrix with rows for different energies and columns for different z,
	ebl=ebl [ lambda, z ]
	"""
	self.model = model
	# Get EBL from data file
	if model == 'kneiske': 
	    if file_name == 'None':
		file_name = '/afs/desy.de/user/m/meyerm/projects/blazars/EBLmodelFiles/mfn_kneiske.dat.gz'
	    f = gzip.open(file_name)
	    self.z, self.ETeV, self.mfn = pickle.load(f)
	else:
	    raise ValueError("Unknown model chosen! Only Kneiske implemented at this time")

	# Interpolation
	self.mfnSpline = BiSpline(np.log(self.ETeV),self.z,np.log(self.mfn),kx=2,ky=2)

	# Max z values
	self.z_max, self.z_min = max(self.z), min(self.z)

    def get_mfn(self,z,ETeV):
	"""
	Return Mean free path in m for redshift z and Energy (TeV) from BSpline Interpolation
	"""
	return np.exp(self.mfnSpline(np.log(ETeV),z)[0,0])

#---------------------------------------------------------------------------
from a3p.constants import *

def calc_ebl_attenuation(z_end, E_TeV, mfn, h=0.7, W_m=0.3, W_l=0.7) :

    int_steps_z = 21

    z_arr = np.linspace(0., z_end, int_steps_z, endpoint=True)
    z_int_arr = np.zeros(int_steps_z)

    # Assign functions/constants to variables to speed up loop
    int_func = scipy.integrate.simps
    #int_func = integrate.trapz
    msqrt = math.sqrt
    get_mfn = mfn.get_mfn

    for i,z in enumerate(z_arr):
	zp1 = z + 1.
	cos = zp1 * msqrt(zp1 * zp1 * (1. + W_m * z) - z * (2. + z) * W_l)
	z_int_arr[i] = 1./get_mfn(z,E_TeV)/cos
    return SI_c * int_func(z_int_arr, z_arr) * 1E9 * astro.yr_s / (astro.WMAP3_H0100 * h)

#===========================================================================
