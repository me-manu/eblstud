import numpy as np
from numpy import log,vstack,dstack,linspace,exp,ma,ones,sum
import math
from scipy import integrate
from scipy import Inf
from eblstud.astro.cosmo import *
from ebltable.tau_from_model import OptDepth
from eblstud.tools.iminuit_fit import pl,lp
from eblstud.misc.constants import *
import logging

class Cascade(object):
    """
    Class to calculate cascade emission 
    following Dermer et al. (2011), Tavecchio et al. (2011), Meyer et al. (2012).
    """
    def __init__(self,**kwargs):
	"""
	Initialize the Cascade class.

	kwargs
	------
	BIGMF:		float, 
			intergalactic magnetic field strength 
			in 10^-15 G (default = 1.)
	cohlnth:	float, 
			intergalactic magnetic field coherence length in Mpc
			(default = 1.)
	tmax:		float, 
			time in years for which AGN has been 
			emitting gamma-rays (default = 10.)
	EmaxTeV:	float, 
			maximum energy of primary gamma-ray emission 
			in TeV (default = 50.)
	EminTeV:	float, 
			minimum energy of primary gamma-ray emission 
			in TeV (default = 0.01)
	eblModel:	string,
			ebl model string (default = 'franceschini')
	bulkGamma:	float, 
			gamma factor of bulk plasma in jet (default = 1.)
	intSpec:	function pointer to intrinsic AGN spectrum, 
			call signature: intSpec(Energy (TeV), intSpecPar) 
			in 1 / TeV / cm^2 / s
			(default= power law)
	intSpecPar:	dictionary, 
			intrinsic spectral parameters 
			(default: 
			{'Prefactor': 1e-10, 'Scale': 100 GeV, 'Index': -2.})
	zSource:	float, source redshift (default = 0.1)
	"""
	# --- setting the defaults
	kwargs.setdefault('BIGMF',1.)
	kwargs.setdefault('cohlnth',1.)
	kwargs.setdefault('zSource',0.1)
	kwargs.setdefault('tmax',10.)
	kwargs.setdefault('eblModel','franceschini')
	kwargs.setdefault('bulkGamma',1.)
	kwargs.setdefault('EmaxTeV',15.)
	kwargs.setdefault('EminTeV',0.01)
	kwargs.setdefault('intSpec',lambda e,**par: pl(e, **par))
	kwargs.setdefault('intSpecPar',
			    {'Prefactor': 1e-10, 'Scale': 100. , 'Index': -2.}
			    )
	
	# ------------------------
	self.__dict__.update(kwargs)

	self.tau = OptDepth.readmodel(model = self.eblModel)

	# energy (in GeV) of CMB photon upscattered by electron produced in EBL
	self.E2eps = lambda ETeV: 0.63*ETeV**2.
	# pair production with primary gamma-ray of energy E in TeV
	self.eps2E = lambda epsGeV: 1.26*np.sqrt(epsGeV)	
	# energy (in TeV) of primary gamma-ray for 
	# upscattered CMB photon of energy eps (GeV)
	# IC cooling length for primary gamma rays. 
	# See e.g. Meyer et al. (2016), Eq. 3
	self.D_IC_Mpc= lambda epsGeV: 0.7 / self.eps2E(epsGeV)

	return
    def _F_IC_T(self,x):
	"""
	Calculate inverse compton kernel in Thomson regime.
    
	Parameter
	---------
	x:	array with x = eps / (4. * E * gamma ** 2.)
    
	Returns
	-------
	array with IC kernel
	"""
	return 2.*x *np.log(x) + x + 1. - 2.*x**2.


    def weight(self,epsGeV):
	"""
	Calculate weighting factor to account for fact if 
	IC cooling length is larger than B field coherence length

	Parameters
	----------
	epsGeV:		n-dim array, energy of upscattered CMB photons
	"""
	D_IC = self.D_IC_Mpc(epsGeV)
	m = D_IC > self.cohlnth
	result = np.ones(epsGeV.shape)
	result[m] = np.sqrt(self.cohlnth / D_IC[m])
	return result

    def gammaEngine(self,epsGeV):
	"""
	Returns minimum gamma factor for IC scattering (Thomson limit integration)
	for cascade emission. 

	gammaEngine is the gamma factor to which e^+e^- pairs have cooled
	if source is running for certain time.
	It also depends on the optical depth of the primary emission, 
	the B field, source redshift and energy
	upscattered photon.

	Parameters
	----------
	epsGeV:		n-dim array, energy of upscattered CMB photons

	Returns
	-------
	n-dim array with minimum gamma factors.

	Notes
	-----
	See Dermer et al. (2011) Eq. 7 and 4
	"""
	# Compute the luminosity distance d_L in units 100 Mpc 
        # and approximate mean free path
	# for pair production by d_L / tau 
	d_L = LumiDistance(self.zSource) / Mpc2cm / 100.
	# Compute mean free path in 100 Mpc 
	# optical depth needs to be interpolation pointer with right redshift
	d_L /= self.tau.opt_depth(self.zSource,self.eps2E(epsGeV))
	# Also compute the IC cooling length, to compare to the coherence length
	# of the B field. Use this to calculate the weigthing factor. 
	w = self.weight(epsGeV)

	return 1.18e8 * np.sqrt(np.sqrt(d_L/self.tmax)*self.BIGMF*w) / \
			(1. + self.zSource)

    def gammaDeflect(self,epsGeV):
	"""
	Returns minimum gamma factor for IC scattering (Thomson limit integration)
	for cascade emission. Denotes the gamma factor when e^+e^- pairs 
	are deflected outside the illuminated cone with opening angle 1/BulkGamma.

	Parameters
	----------
	epsGeV:		n-dim array, energy of upscattered CMB photons

	Returns
	-------
	n-dim array with minimum gamma factors.

	Notes
	-----
	"""
	# Also compute the IC cooling length, to compare to the coherence length
	# of the B field. Use this to calculate the weigthing factor. 
	w = self.weight(epsGeV)
	return 1.08e6*np.sqrt(self.BIGMF*w*self.bulkGamma)/\
		(1. + self.zSource)**2. * np.ones(epsGeV.shape[0])
	# Isotropy:
	#return 1.08e6*np.sqrt(B_IGMF_15/np.pi)/(1. + z)**2.

    def gammaCMB(self,epsGeV):
	"""
	Returns minimum gamma factor for IC scattering (Thomson limit integration)
	for scattering up CMB photons. 

	Parameters
	----------
	epsGeV:		n-dim array, energy of upscattered CMB photons

	Returns
	-------
	n-dim array with minimum gamma factors.

	Notes
	-----
	"""
	return np.sqrt(epsGeV*1e9/4./E_CMB)

    def gammaMin(self,epsGeV):
	"""
	Determines the minimum gamma factor for IC scattering 
	(Thomson limit integration)
	for scattering up CMB photons by choosing 
	the maximum of gammaCMB, gammaEngine, and gammaDeflect.

	Parameters
	----------
	epsGeV:		n-dim array, energy of upscattered CMB photons

	Returns
	-------
	n-dim array with minimum gamma factors.

	Notes
	-----
	"""
	g		= self.gammaCMB(epsGeV)
	g		= np.vstack((g,self.gammaEngine(epsGeV)))
	g		= np.vstack((g,self.gammaDeflect(epsGeV)))
	return np.max(g, axis = 0)

    def cascadeSpec(self,
			epsGeV,
			gmin = None, 
			gsteps = 25, 
			Esteps = 25, 
			epssteps = 25, 
			epsmin = 1e-10, 
			epsmax = 1e1,
			intSpectrum = True
			):
	"""
	Calculate cascade spectrum.  

	Parameters
	----------
	epsGeV:		n-dim array, energy of upscattered CMB photons, in GeV

	kwargs
	------
	gmin:	n-dim array with minimum gamma-factors. 
		If none, calculated from gammaMin function (default = None)
	gsteps:	int, number of steps for gamma integration (default = 20)
	Esteps:	int, number of steps for energy integrations 
	epsmin: float, minimum energy of CMB integration, in eV (default = 1e-10)
	epsmax: float, maximum energy of CMB integration, in eV (default = 1e1)
	intSpectrum: bool, 
		if True, provided spectrum assumed to be the intrinsic spectrum, 
		    otherwise, assumed to be observed spectrum

	Returns
	-------
	n-dim array with cascade spectrum in 1/cm^2/s/eV 

	Notes
	-----
	"""
	if gmin == None:
	    logGmin = log(self.gammaMin(epsGeV))
	else:
	    logGmin = log(gmin)

	# maximum gamma factor 
	# corresponding to max prim. energy
	#logGmax = log(self.EmaxTeV * 1e12 / (2. * M_E_EV)) * 0.999 
	logGmax = log(1e8)

	# check if maximum gamma factor is larger than minimum gamma factor
	if np.all(logGmin >= logGmax):
	    wString = '*** all minimum gamma values are larger than the maximum gamma value:'
	    logging.warning(wString)
	    logging.warning('*** gmax: {0:.3e}'.format(
			exp(logGmax)))
	    logging.warning('*** max(gammaMin): {0:.3e}'.format(
			np.max(self.gammaMin(epsGeV))))
	    logging.warning('*** max(gammaEngine): {0:.3e}'.format(
		np.max(self.gammaEngine(epsGeV))))
	    logging.warning('*** max(gammaDeflect): {0:.3e}'.format(
			np.max(self.gammaDeflect(epsGeV))))
	    logging.warning('*** max(gammaCMB): {0:.3e}'.format(
			np.max(self.gammaCMB(epsGeV))))
	    return np.zeros(epsGeV.shape) 

	# mask out regions that where gmin > gmax
	epsGeV = ma.array(epsGeV, mask = logGmin >= logGmax)
	logGmin = ma.array(logGmin, mask = epsGeV.mask)

	# define the energy arrays for integration
	# outer integral runs over log gamma (one log gamma for each cascade energy)
	# inner integral runs over primary energy spectrum
	# second inner integral runs over energy of CMB spectrum
	logG = []
	logEprim  = []
	logEpsCMB = []
	tauPrim  = []

	logG = ma.zeros((epsGeV.shape[0],gsteps))
	logG.mask = np.zeros(logG.shape, dtype = bool)
	for ilg, logg in enumerate(logGmin): # first loop over minimum gamma factors
	    logG.mask[ilg] = logGmin.mask[ilg] * np.ones(gsteps, dtype = bool)
	    if not logGmin.mask[ilg]:
		# check if corresponding minimum energy is larger than max energy
		if logg + log(2. * M_E_EV * 1e-12) > log(self.EmaxTeV):
		    logG.mask[ilg] = np.ones(gsteps, dtype = bool)
		else:
		    logG[ilg] =  np.linspace(logg, logGmax, gsteps )

	# calculate the lower bound for the first inner integral for the injected spectrum
	# in TeV
	logEmin = log(self.EminTeV) * np.ones(logG.shape)
	logEprimMinTeV = logG + log(2. * M_E_EV * 1e-12) 
	logEprimMinTeV = logEprimMinTeV * (logEprimMinTeV > logEmin) + \
				logEmin * (logEprimMinTeV <= logEmin)

	# generate three dim arrays for the intergral over the primary spectrum
	# also generate the optical depths for the injected energies
	logEprim = ma.zeros((epsGeV.shape[0], gsteps, Esteps))
	logEprim.mask = np.zeros(logEprim.shape, dtype = bool)
	tauPrim = ma.zeros((epsGeV.shape[0], gsteps, Esteps))
	tauPrim.mask = np.zeros(tauPrim.shape, dtype = bool)

	for ilE, logE in enumerate(logEprimMinTeV):
	    for jlE, logEE in enumerate(logE):
		logEprim.mask[ilE,jlE] = logEprimMinTeV.mask[ilE,jlE] * \
					    np.ones(Esteps, dtype = bool)

		if not (logEprimMinTeV.mask[ilE,jlE] or logEE > log(self.EmaxTeV)):
		    logEprim[ilE,jlE] = np.linspace(logEE, log(self.EmaxTeV), Esteps)
		    tauPrim[ilE,jlE] = self.tau.opt_depth(
				    self.zSource,exp(logEprim[ilE,jlE]))

	# if all masks are true (gmin > gmax) || (ETeVmin > ETeVmax)
	# return zero
	if np.all(logEprim.mask):
	    logging.warning("All (gmin > gmax) || (ETeVmin > ETeVmax), returning 0.")
	    return np.zeros(epsGeV.shape)

	tauPrim.mask = logEprim.mask
	# generate three dim arrays for the intergral over the CMB spectrum
	logEpsCMB = ma.zeros((epsGeV.shape[0], gsteps, epssteps))
	logEpsCMB.mask = np.zeros(logEpsCMB.shape, dtype = bool)

	x = ma.zeros((epsGeV.shape[0], gsteps, epssteps))

	for ilg, lg in enumerate(logG): 
	    for jlg, lgg in enumerate(lg): 

		logEpsCMB[ilg,jlg] = np.linspace(log(epsmin), log(epsmax), epssteps)
		x[ilg,jlg] = epsGeV[ilg] * 1e9 / 4. / exp(logEpsCMB[ilg,jlg]) / \
				    exp(2. * logG[ilg,jlg])

	
	x.mask = logEpsCMB.mask | (x >= 1)

	# calculate kernel for CMB integral ---------------------- #
	# the rollaxis calls do cast the arrays into the right shapes to 
	# be multiplied
	
	# log(x) will return warning for masked entries
	kernelCMB = self._F_IC_T(x) * 4. * \
			nphotCMBarray(exp(logEpsCMB)) / exp(logEpsCMB)

	kernelCMB *= exp(logEpsCMB)
	# rollaxis needed, since logG and kernelCMB have different dimensions
	kernelCMB = np.rollaxis(
			np.rollaxis(
			    np.rollaxis(kernelCMB,2) * exp(2. * logG), 2
			    ), 2)


	# Do the integration over CMB energy, with rollaxis this is axis 0 at the moment
	kernelGamma = simps(kernelCMB, logEpsCMB, axis = 2)


	# calculate kernel for injected source integral ------------------- #
	kernelInjSpec = self.intSpec(exp(logEprim), **self.intSpecPar)
	if intSpectrum: # provided spectrum is the intrinsic one
	    kernelInjSpec *= 1. - exp(-tauPrim)
	else: # provided spectrum is the observed one
	    kernelInjSpec *= exp(tauPrim) - 1.
	kernelInjSpec *= exp(logEprim) # account for log integration


	kernelGamma *= simps(kernelInjSpec, logEprim, axis = 2) / exp(logG * 5.)
	mask = np.isnan(kernelGamma)

	kernelGamma = ma.array(kernelGamma, mask = mask)
	logG = ma.array(logG, mask = mask | logG.mask)

	#logging.info('new {0}'.format(kernelGamma))
	#logging.info('new {0}'.format(kernelGamma.shape))
	#logging.info('new {0}'.format(logG))
	#logging.info('new {0}'.format(logG.shape))
	
	result = np.zeros(epsGeV.shape)
	for ikG, kG in enumerate(kernelGamma):
	    if np.all(kG.mask): continue
	    result[ikG] = simps(kG.compressed(), logG[ikG].compressed())
	result *= M_E_EV / U_CMB * 9. / 64.
	
	return result

    def binBybinCascade(self,EeVinj_lbounds, EeVinj_centers, EeVobs_centers, **kwargs):
    	"""
	Calculate the cascade for energy bins of the injected 
	spectrum.

	Parameters
	----------
	EeVinj_lbounds: n+1 dim - ndarray,
		array with energy bin bounds, in eV. 
		Injected energies.
	EeVinj_lbounds: n dim - ndarray,
		array with energy bin centers, in eV. 
		Injected energies.
	EeVobs_centers: m dim - ndarray,
		array with center of energy bins for which cascade is
		calculated. In eV.

	kwargs
	------
	see cascadeSpec function

	Returns
	-------
	Tuple with injected, observed primary, observed cascade flux in 1/eV/s/cm**2.
	"""
	Emin, Emax = self.EminTeV, self.EmaxTeV
	cascFlux = np.zeros((EeVinj_centers.shape[0], EeVobs_centers.shape[0]))
	for i,elb in enumerate(EeVinj_lbounds[:-1]):
	    self.EminTeV = elb * 1e-12
	    self.EmaxTeV = EeVinj_lbounds[i + 1] * 1e-12
	    cascFlux[i] = self.cascadeSpec(EeVobs_centers * 1e-9, **kwargs)

	self.EminTeV, self.EmaxTeV = Emin, Emax
	injFlux = self.intSpec(EeVinj_centers * 1e-12, **self.intSpecPar)
	primFlux = injFlux * exp(-self.tau.opt_depth(self.zSource, 
					EeVinj_centers * 1e-12))

	return injFlux, primFlux, cascFlux

# eps in GeV for TeV photons converted to e^+e^- pairs 
# and IC scattering on CMB photons in the Thomson regime
E2eps = lambda ETeV: 0.63*ETeV**2.
# Inverse of the above:
eps2E = lambda epsGeV: 1.26*np.sqrt(epsGeV)


# -------------------------------------------------------------------------------------------------#
# --- OLD OUTDATED CODE. USE THE ABOVE CLASS AND FUNCTIONS. ---------------------------------------#
# -------------------------------------------------------------------------------------------------#

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

def CascadeSpecSimps(eps1, gmin, tau, specparams = (1e-10,-2.,10.), EmaxTeV = 15. ):
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
	#Emax = math.log(5e1)
	Emax = math.log(EmaxTeV)
	if Emin > Emax:
	    gamma_kern_array[i] = 0.
	else:
	    log_E_array = np.linspace(Emin,Emax,E_steps)
	    E_inject_array = np.empty((len(log_E_array),))
	    for j,e in enumerate(log_E_array):
		E = np.exp(e)
		E_inject_array[j] = E  ** G * math.exp(-(E/Ebr) ** 10.)*( np.exp(tau(E)) - 1. )
		#E_inject_array[j] = E ** G*( np.exp(tau(E)) - 1. )
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

    #logging.info('old {0} {1}'.format(i,gamma_kern_array))
    #logging.info('old {0} {1}'.format(i,log_gamma_array))
    result = integrate.simps(gamma_kern_array,log_gamma_array)
    # result is in photon / cm^3 / eV / cm^2 / s
    # Convert to photons / cm^2 / eV by multiplying with m_e c^2 / u_cmb [cm^3]
    result *= M_E_EV / U_CMB * 9. / 64. * N0
    return result

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
