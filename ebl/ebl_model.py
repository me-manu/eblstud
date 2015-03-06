"""
Class to read EBL models of Kneiske & Dole 2010 and Franceschini et al. 2008

History of changes:
Version 1.1
- Created 25th November 2010
- Updated 16th November 2011, now only for nuI_nu model files
  and included integration for mean free path and optical depth

"""

__version__ = 1.1
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

# ================================================================================#
# Imports
import numpy as np
import math
import scipy.integrate
from scipy.interpolate import RectBivariateSpline as BiSpline
import scipy.interpolate
from eblstud.misc.constants import *
from os.path import *

# ================================================================================#

def nuInu_Micron_2_JPhotDens(l,nuInu):
    """
    Convert 2D nuInu Matrix into 2D photon density matrix [1/[m^3 J]] and
    convert wavelength into energy
    """
    e_J = SI_h*SI_c / (l * 1e-6)
    n = (4.*PI / SI_c / e_J**2. * nuInu.transpose()).transpose()*1e-9
    return np.log(e_J), np.log(n)

# -- EBL Model from 2D Matrix ----------------------------------------------------------------#
class EBLModel(object):
    def __init__(self,path='/afs/desy.de/user/m/meyerm/projects/blazars/EBLmodelFiles/',model='kneiske'):
	"""
	Read EBL or Optical Depth Data from 2D Matrix

	First Row: Redshift z
	First Column:  Wavelength
	Matrix Entries lambda I_lambda(lambda,z) 
	First Index: Energy (rows)
	Second Index: Redshift
	self.eblis 2D Matrix with rows for different wavelengths and columns for different z,
	ebl=ebl [ lambda, z ]
	"""
	self.model = model
	try:
	    ebl_file_path = os.environ['EBL_FILE_PATH']
	except KeyError:
	    warnings.warn("The EBL File environment variable is not set! Using {0} as path instead.".format(path), RuntimeWarning)
	    ebl_file_path = path
	# Get EBL from data file
	if model == 'kneiske': 
	    if file_name == 'None':
		file_name = join(ebl_file_path,'nuInu_kneiske.dat')
	    data = np.loadtxt(file_name)
	    self.z = data[0,1:]
	    self.ebl= data[1:,1:]
	    if model == 'kneiske':
		self.l = data[1:,0]
	elif model == 'cuba': 
	    if file_name == 'None':
		file_name = join(ebl_file_path,'CUBA_UVB.dat')
	    data = np.loadtxt(file_name)
	    self.z = data[0,1:]
	    self.l = data[1:,0] * 1e-4	# in microns
	    self.ebl= data[1:,1:] * SI_c / (self.l * 1e-6) # in nu I nu
	else:
	    raise ValueError("Unknown model chosen! Only Kneiske implemented at this time")

	# Convert nuInu to photons / m^-3 and lambda to energy [eV] in logscale
	self.e_J,self.n = nuInu_Micron_2_JPhotDens(self.l,self.ebl)
	# Revert arrays for Interpolation
	self.e_J,self.n = np.flipud(self.e_J),np.flipud(self.n)

	# Interpolation
	self.eblSpline = BiSpline(np.log(self.l),np.log(self.z),np.log(self.ebl),kx=2,ky=2)
	self.nSpline = BiSpline(self.e_J,np.log(self.z),self.n,kx=2,ky=2)

	# Get min and max
	self.l_min, self.l_max = min(self.l), max(self.l)
	self.e_J_min,self.e_J_max = np.exp(min(self.e_J)),np.exp(max(self.e_J))
	self.e_min_J,self.e_max_J = np.exp(min(self.e_J)),np.exp(max(self.e_J))

	return


    def nuInu(self,z,micron):
	"""
	Return EBL density in nu I nu for redshift z and Wavelngth (micron) from BSpline Interpolation
	"""
	if z < self.z[0]:
	    return np.exp(self.eblSpline(np.log(micron),np.log(self.z[0]))[0,0])
	return np.exp(self.eblSpline(np.log(micron),np.log(z))[0,0])

    def get_n_ll(self,z,log_e_J):
	"""
	Return EBL photon density in m^-3 for redshift z and energy (J) from BSpline Interpolation
	in log log
	"""
	if z < self.z[0]:
	    return self.nSpline(log_e_J,np.log(self.z[0]))[0,0] + 3.*np.log(1. + z)
	return self.nSpline(log_e_J,np.log(z))[0,0] + 3.*np.log(1. + z)

    def get_n(self,z,e_J):
	"""
	Return EBL photon density in m^-3 for redshift z and energy (J) from BSpline Interpolation
	"""
	#if z < self.z[0]:
	    #return np.exp(self.nSpline(np.log(e_J),np.log(self.z[0]))[0,0])*(1. + z)**3.
	return np.exp(self.nSpline(np.log(e_J),np.log(z))[0,0])*(1. + z)**3.

# --------------------------------------------------------------------------------------------#
def calc_mean_free_path(z, E_TeV, ebl):
    """
    Calculate mean free path in meters
    for photons with Energy E_TeV at redshift z 
    traversing the EBL given by ebl.
    Orientated on M. Raue's calc_ebl_attenuation function
    """
    E_J = E_TeV * 1e12 * SI_e

    int_steps_mu, int_steps_log_e = 21,301

    mu_arr = np.linspace(-1. + 1E-6, 1. - 1E-6, int_steps_mu, endpoint=True)
    mu_int_arr = np.zeros(int_steps_mu)

    # Functions for faster lookup
    int_simps = scipy.integrate.simps
    get_n = ebl.get_n
    log = math.log
    linspace,zeros, exp,sqrt,nplog = np.linspace,np.zeros,np.exp,np.sqrt,np.log
    M, TCS = M_E_EV * SI_e, SI_tcs
    e_max,e_min = ebl.e_J_max,ebl.e_J_min

    zp1 = 1. + z
    for imu, m in enumerate(mu_arr):
	e_thr = 2.*M*M/E_J/(1. - m)/zp1
	log_e_thr = log(e_thr)

	log_e_int_arr = zeros(int_steps_log_e)
	log_e_arr = linspace(log_e_thr+1e-8, log_e_thr+log(1e5),int_steps_log_e,endpoint = True)
	e = exp(log_e_arr)
	# numpy with mask - fastest
	mask = (e > e_min) * (e < e_max)
	if np.any(mask):
	    b = sqrt(1. - e_thr / e[mask])
	    bb = b * b
	    r = (1. - bb) * (2. * b * (bb - 2.) + (3. - bb * bb) * nplog((1. + b) / (1. - b)))
	    # This step takes a long time :-(
	    log_e_int_arr[mask] = r * np.array(map(lambda E: get_n(z,E),e[mask])) * e[mask]
	mu_int_arr[imu] = int_simps(log_e_int_arr,log_e_arr) * (1. - m)
	if math.isnan(mu_int_arr[imu]) :
	    mu_int_arr[imu] = 0.

    return 1./(int_simps(mu_int_arr,mu_arr) * 3. / 16. * TCS / 2.)# / (1. + z))

#---------------------------------------------------------------------------
from a3p.constants import *

def calc_ebl_attenuation(z_end, E_TeV, ebl, h=0.7, W_m=0.3, W_l=0.7) :
    """
    Calculates the optical depth for VHE gamma-rays for a given redshift, energy, and EBL density.
    Orientated on M. Raue's calc_ebl_attenuation function
    """

    E_J = E_TeV * 1e12 * SI_e

    int_steps_z, int_steps_mu, int_steps_log_e = 21, 21, 301

    z_arr = np.linspace(0., z_end, int_steps_z, endpoint=True)
    mu_arr = np.linspace(-1. + 1E-6, 1. - 1E-6, int_steps_mu, endpoint=True)
    log_e_arr = np.zeros(int_steps_log_e)
    z_int_arr = np.zeros(int_steps_z)
    mu_int_arr = np.zeros(int_steps_mu)
    log_e_int_arr = np.zeros(int_steps_log_e)

    # Assign functions/constants to variables to speed up loop
    int_func = scipy.integrate.simps
    #int_func = integrate.trapz
    log,msqrt = math.log,math.sqrt
    linspace,zeros,nplog, sqrt,exp = np.linspace,np.zeros,np.log,np.sqrt,np.exp
    get_n = ebl.get_n
    e_max,e_min = ebl.e_J_max,ebl.e_J_min
    me = M_E_EV * SI_e 

    for i, z in enumerate(z_arr) :
        zp1 = z + 1.
        for j, mu in enumerate(mu_arr) :
            e_thr = 2. * me*me / E_J / (1. - mu) / zp1
            log_e_thr = log(e_thr)
            log_e_arr = linspace(log_e_thr + 1E-8, log_e_thr + 12., int_steps_log_e,
                                   endpoint=True)


            # numpy with mask - fastest
            log_e_int_arr = zeros(int_steps_log_e)
            e = exp(log_e_arr)
            m = (e < e_max) * (e > e_min)
            if np.any(m) :
                b = sqrt(1. - e_thr / e[m])
                bb = b * b
                r = (1. - bb) * (2. * b * (bb - 2.) + (3. - bb * bb) * np.log((1. + b) / (1. - b)))
		# This step takes a long time :-(
                log_e_int_arr[m] = r * np.array(map(lambda E: get_n(z,E),e[m])) * e[m]

            mu_int_arr[j] = int_func(log_e_int_arr, log_e_arr) * (1. - mu) / 2.
            if math.isnan(mu_int_arr[j]) :
                mu_int_arr[j] = 0.
        cos = zp1 * msqrt(zp1 * zp1 * (1. + W_m * z) - z * (2. + z) * W_l)
        z_int_arr[i] = int_func(mu_int_arr, mu_arr) / cos
    return 3. / 16. * SI_tcs * SI_c * int_func(z_int_arr, z_arr) * 1E9 * astro.yr_s / (astro.WMAP3_H0100 * h)

#===========================================================================
