import numpy as np
import math
from scipy import integrate
from scipy import Inf
from eblstud.astro.cosmo import *

# eps in GeV for TeV photons converted to e^+e^- pairs 
# and IC scattering on CMB photons in the Thomson regime
E2eps = lambda ETeV: 0.63*ETeV**2.
# Inverse of the above:
eps2E = lambda epsGeV: 1.26*np.sqrt(epsGeV)

# Computation of Lower Limits for IC integration over gamma ---------------------------------------#
def GammaMin(t_yr,B_IGMF_15,z,tau,eps_GeV,BulkGamma):
#    return max(GammaEngine(t_yr,B_IGMF_15,z,tau,eps_GeV),GammaDeflect(B_IGMF_15,BulkGamma,z),
#		GammaCMB(eps_GeV))
    return max(GammaEngine(t_yr,B_IGMF_15,z,tau,eps_GeV),
		GammaCMB(eps_GeV))
#    return max(GammaDeflect(B_IGMF_15,BulkGamma,z),
#		GammaCMB(eps_GeV))
#    return GammaEngine(t_yr,B_IGMF_15,z,tau,eps_GeV)
#    return GammaDeflect(B_IGMF_15,BulkGamma,z)
#    return GammaCMB(eps_GeV))

def GammaEngine(t_yr,B_IGMF_15,z,tau,eps_GeV):
    """
    returns minimum gamma factor for IC (Thomson limit integration)
    for cascade emission. 
    GammaEngine is the gamma factor to which e^+e^- pairs have cooled
    if source is running for a time t in years.
    It also depends on the optical depth of the primary emission, 
    where the cascade emission energy is given by eps in GeV.
    """
    # Compute the luminosity distance d_L in units 100 Mpc and approximate mean free path
    # for pair production by d_L / tau 
    d_L = LumiDistance(z) / Mpc2cm / 100.
    # Compute mean free path in 100 Mpc 
    # optical depth needs to be interpolation pointer with right redshift
    d_L /= tau(eps2E(eps_GeV))
    return 1.18e8 * np.sqrt(np.sqrt(d_L/t_yr)*B_IGMF_15) / (1. + z)

def GammaDeflect(B_IGMF_15,BulkGamma,z):
    """
    returns minimum gamma factor for IC (Thomson limit integration)
    for cascade emission. Denotes the gamma factor when e^+e^- pairs 
    are deflected outside the illuminated cone with opening angle 1/BulkGamma.
    See Dermer et al. 2011.
    Inter galactic B field is in 10^-15 G
    """
    return 1.08e6*np.sqrt(B_IGMF_15*BulkGamma)/(1. + z)**2.
    # Isotropy:
    #return 1.08e6*np.sqrt(B_IGMF_15/np.pi)/(1. + z)**2.


def GammaCMB(eps_GeV):
    """
    returns minimum gamma factor for IC (Thomson limit integration)
    for scattering on CMB photons. 
    eps is in GeV
    """
    return np.sqrt(eps_GeV*1e9/4./E_CMB)
# -------------------------------------------------------------------------------------------------#

def CascadeSpec(eps1, gmin, tau, specparams = (1e-10,-2.,10.) ):
    """
    Returns Cascade spectrum in 1/cm^2/s/eV at energy eps1 in eV 
    """
    N0,G,Ebr = specparams
    gmin = math.log(gmin)
    gmax = math.log(1e20)
    if gmin >= gmax:
	return 0.
    result = integrate.quad(CascadeKernGamma,gmin,gmax,args = (eps1, tau, N0, G, Ebr) )[0]
    # result is in photon / cm^3 / eV / cm^2 / s
    # Convert to photons / cm^2 / eV by multiplying with m_e c^2 / u_cmb [cm^3]
    result *= M_E_EV / U_CMB * 9. / 64.
    return result

def CascadeKernGamma(gamma, eps1, tau, N0, G, Ebr):
    gamma = math.exp(gamma)
    # E in TeV
    Emin = math.log(2.*M_E_EV*1e-12*gamma)
    Emax = math.log(5e1)
    if Emin >= Emax:
	return 0.
    # result of integration in units 1/cm^2/s
    result = integrate.quad(CascadeKernKernInjection,Emin,Emax,args = (tau , N0, G, Ebr ))[0]

    # eps in eV
    epsmin = math.log(1e-10)
    epsmax = math.log(1e1)
    if epsmin >= epsmax:
	return 0.
    # result of integration in units 1/cm^3/eV
    result *= integrate.quad(CascadeKernKernEps,epsmin,epsmax,args = (gamma,eps1) )[0]

    result /= gamma ** 6.
    result *= gamma
    return result


def CascadeKernKernInjection(E, tau , N0, G, Ebr ):
    """
    Log integration over absorbed intrinsic blazar spectra. E is in TeV
    Observed spectra described by power law with super exponential cut off.
    The optical depth is given as an interpolated function tau(E)
    """
    E = math.exp(E)
    result = N0 * E ** G * math.exp(-(E/Ebr) ** 10.)
    result *= ( math.exp(tau(E)) - 1. )
    result *= E
    return result

F_IC_T = lambda x: 2.*x *np.log(x) + x + 1. - 2.*x**2.

def CascadeKernKernEps(eps, gamma, eps1 ):
    eps = math.exp(eps)
    x = eps1 / (4. * eps * gamma ** 2.)
    if x >= 1.:
	return 0.
    result = F_IC_T(x)
    result *= 4. * gamma**2. * eps
    result *= nphotCMB(eps) / eps ** 2.
    result *= eps
    return result

def CascadeSpecSimps(eps1, gmin, tau, specparams = (1e-10,-2.,10.) ):
    """
    Returns Cascade spectrum in 1/cm^2/s/eV at energy eps1 in eV 
    """
    N0,G,Ebr = specparams
    gmin = math.log(gmin)
    gmax = math.log(1e8)

    gamma_steps = 16
    E_steps	= 16
    eps_steps	= 26

    if gmin >= gmax:
	return 0.

    log_gamma_array = np.linspace(gmin,gmax,gamma_steps)
    gamma_kern_array = np.empty((len(log_gamma_array),))
    for i,g in enumerate(log_gamma_array):
	Gamma = np.exp(g)
	Emin = math.log(2.*M_E_EV*1e-12*Gamma)
	Emax = math.log(5e1)
	if Emin > Emax:
	    gamma_kern_array[i] = 0.
	else:
	    log_E_array = np.linspace(Emin,Emax,E_steps)
	    E_inject_array = np.empty((len(log_E_array),))
	    for j,e in enumerate(log_E_array):
		E = np.exp(e)
		E_inject_array[j] = E ** G * math.exp(-(E/Ebr) ** 10.)*( math.exp(tau(E)) - 1. )
		E_inject_array[j] *= E

	    gamma_kern_array[i] = integrate.simps(E_inject_array,log_E_array)

	    epsmin = math.log(1e-10)
	    epsmax = math.log(1e1)
	    log_eps_array = np.linspace(epsmin,epsmax,eps_steps)
	    eps_kern_array = np.empty((len(log_eps_array),))
	    for j,e in enumerate(log_eps_array):
		E = np.exp(e)
		x = eps1 / (4. * E * Gamma ** 2.)
		if x >= 1.:
		    eps_kern_array[j] = 0.
		else:
		    eps_kern_array[j]= F_IC_T(x)* 4. * Gamma**2. *nphotCMB(E) / E
		    eps_kern_array[j]*= E 

	    gamma_kern_array[i] *= integrate.simps(eps_kern_array,log_eps_array)
	    gamma_kern_array[i] /= Gamma ** 6.
	    gamma_kern_array[i] *= Gamma

    result = integrate.simps(gamma_kern_array,log_gamma_array)
    # result is in photon / cm^3 / eV / cm^2 / s
    # Convert to photons / cm^2 / eV by multiplying with m_e c^2 / u_cmb [cm^3]
    result *= M_E_EV / U_CMB * 9. / 64. * N0
    return result

from a3p.constants import *
def calc_EBL_mean_free_path(z_end, E_TeV, ebl, h=h, W_m=0.26, W_l=0.74) :
    """
    Calculates the mean free path for pair production on the EBL
    """

    E_J = E_TeV * 1e12 * misc.eV_J
    LOG10 = math.log(10)

    int_steps_mu, int_steps_log10e = 21, 201

    mu_arr = np.linspace(-1., 1., int_steps_mu, endpoint=True)
    log10_e_arr = np.zeros(int_steps_log10e)
    mu_int_arr = np.zeros(int_steps_mu)
    log10_e_int_arr = np.zeros(int_steps_log10e)

    for j, mu in enumerate(mu_arr) :
	e_thr = 2. * (si.c * si.c * si.me) ** 2. / E_J / (1. - mu) / (z_end + 1.)
	log10_e_thr = math.log10(e_thr)
	log10_e_arr = np.linspace(log10_e_thr + 1E-8, log10_e_thr + 5, int_steps_log10e,
                                      endpoint=True)
	for k, log10_e in enumerate(log10_e_arr) :
	    e = 10. ** log10_e
	    b = math.sqrt(1. - e_thr / e)
	    bb = b * b
	    r = (1. - bb) * (2. * b * (bb - 2.) + (3. - bb * bb) * math.log((1. + b) / (1. - b)))
	    log10_e_int_arr[k] = r * ebl.get_n(e, z_end) * e * LOG10
	mu_int_arr[j] = integrate.simps(log10_e_int_arr, log10_e_arr) * (1. - mu) / 2.
	if math.isnan(mu_int_arr[j]) :
	    mu_int_arr[j] = 0.
    result = integrate.simps(mu_int_arr, mu_arr)
    return result
