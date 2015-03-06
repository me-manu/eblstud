import numpy as np
from numpy import log,vstack,dstack,linspace,exp,ma,ones,sum
import math
from scipy import integrate
from scipy import Inf
from eblstud.astro.cosmo import *
import eblstud.ebl.tau_from_model as tfm
from eblstud.tools.iminuit_fit import pl,lp
from eblstud.misc.constants import *

class Cascade(object):
    """
    Class to calculate cascade emission following Dermer et al. (2011), Tavecchio et al. (2011), Meyer et al. (2012).
    """
    def __init__(self,**kwargs):
	"""
	Initialize the Cascade class.

	kwargs
	------
	BIGMF:		float, intergalactic magnetic field strength in 10^-15 G (default = 1.)
	tmax:		float, time in years for which AGN has been emitting gamma-rays (default = 10.)
	EmaxTeV:	float, maximum energy of primary gamma-ray emission in TeV (default = 50.)
	eblModel:	ebl model string (default = 'franceschini')
	bulkGamma:	float, gamma factor of bulk plasma in jet (default = 1.)
	intSpec:	function pointer to intrinsic AGN spectrum, call signature: intSpec(Energy (TeV), intSpecPar) (default= power law)
	intSpecPar:	dictionary, intrinsic spectral parameters (default: {'Prefactor': 1e-10, 'Scale': 100 GeV, 'Index': -2.})
	zSource:	float, source redshift (default = 0.1)
	"""
	# --- setting the defaults
	kwargs.setdefault('BIGMF',1.)
	kwargs.setdefault('zSource',0.1)
	kwargs.setdefault('tmax',10.)
	kwargs.setdefault('eblModel','franceschini')
	kwargs.setdefault('bulkGamma',1.)
	kwargs.setdefault('EmaxTeV',15.)
	kwargs.setdefault('intSpec',lambda e,par: pl(par,e))
	kwargs.setdefault('intSpecPar',{'Prefactor': 1e-10, 'Scale': 100. , 'Index': -2.})
	
	# ------------------------
	self.__dict__.update(kwargs)

	self.tau = tfm.OptDepth(model = self.eblModel)

	self.E2eps = lambda ETeV: 0.63*ETeV**2.			# energy (in GeV) of CMB photon upscattered by electron produced in EBL
								# pair production with primary gamma-ray of energy E in TeV
	self.eps2E = lambda epsGeV: 1.26*np.sqrt(epsGeV)	# energy (in TeV) of primary gamma-ray for upscattered CMB photon of energy eps (GeV)

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


    def gammaEngine(self,epsGeV):
	"""
	Returns minimum gamma factor for IC scattering (Thomson limit integration)
	for cascade emission. 
	gammaEngine is the gamma factor to which e^+e^- pairs have cooled
	if source is running for certain time.
	It also depends on the optical depth of the primary emission, the B field, source redshift and energy
	upscattered photon.

	Parameters
	----------
	epsGeV:		n-dim array, energy of upscattered CMB photons

	Returns
	-------
	n-dim array with minimum gamma factors.

	Notes
	-----
	See Dermer et al. (2011) Eq. 
	"""
	# Compute the luminosity distance d_L in units 100 Mpc and approximate mean free path
	# for pair production by d_L / tau 
	d_L = LumiDistance(self.zSource) / Mpc2cm / 100.
	# Compute mean free path in 100 Mpc 
	# optical depth needs to be interpolation pointer with right redshift
	d_L /= self.tau.opt_depth_array(self.zSource,self.eps2E(epsGeV))[0]
	return 1.18e8 * np.sqrt(np.sqrt(d_L/self.tmax)*self.BIGMF) / (1. + self.zSource)

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
	return 1.08e6*np.sqrt(self.BIGMF*self.bulkGamma)/(1. + self.zSource)**2. * np.ones(epsGeV.shape[0])
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
	Determines the minimum gamma factor for IC scattering (Thomson limit integration)
	for scattering up CMB photons by choosing the maximum of gammaCMB, gammaEngine, and gammaDeflect.

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

    def cascadeSpec(self,epsGeV,gmin = None, gsteps = 20, Esteps = 30, epsmin = 1e-10, epsmax = 1e1):
	"""
	Calculate cascade spectrum.  

	Parameters
	----------
	epsGeV:		n-dim array, energy of upscattered CMB photons, in GeV

	kwargs
	------
	gmin:	n-dim array with minimum gamma-factors. If none, calculated from gammaMin function (default = None)
	gsteps:	int, number of steps for gamma integration (default = 20)
	Esteps:	int, number of steps for energy integrations 
	epsmin: float, minimum energy of CMB integration, in eV (default = 1e-10)
	epsmin: float, maximum energy of CMB integration, in eV (default = 1e1)

	Returns
	-------
	n-dim array with cascade spectrum in 1/cm^2/s/eV 

	Notes
	-----
	"""
	if gmin == None:
	    gmin = log(self.gammaMin(epsGeV))
	else:
	    gmin = log(gmin)

	tauCut = 100.
	Emax = 10.**(self.tau.opt_depth_Inverse(self.zSource,tauCut)) / 1e3
	gmax = log(Emax * 1e12 / (2. * M_E_EV)) * 0.999

	if np.all(gmin >= gmax):
	    print '*** all minimum gamma values are larger than the maximum gamma value:'
	    print '*** gmax: {0:.3e}'.format(exp(gmax))
	    print '*** max(gammaMin): {0:.3e}'.format(np.max(self.gammaMin(epsGeV)))
	    print '*** max(gammaEngine): {0:.3e}'.format(np.max(self.gammaEngine(epsGeV)))
	    print '*** max(gammaDeflect): {0:.3e}'.format(np.max(self.gammaDeflect(epsGeV)))
	    print '*** max(gammaCMB): {0:.3e}'.format(np.max(self.gammaCMB(epsGeV)))
	    return np.ones(epsGeV.shape) * -1
	    

	# mask out regions that where gmin < gmax
	gmin = gmin[gmin < gmax]
	epsGeV = epsGeV[gmin < gmax]

	for i,g in enumerate(gmin):
	    if not i:
		logGamma = np.linspace(g,gmax,gsteps)
		for j,lgi in enumerate(logGamma):
		    if not j:
			logEarray	= linspace(log(2. * M_E_EV * 1e-12) + lgi, log(Emax), Esteps)
			logEpsArray	= linspace(log(epsmin), log(epsmax),Esteps)
			tauArray	= self.tau.opt_depth_array(self.zSource,exp(logEarray))[0]
			tauArray[tauArray > tauCut] = tauCut * ones(sum(tauArray > tauCut))
		    else:
			lE		= linspace(log(2. * M_E_EV * 1e-12) + lgi, log(Emax), Esteps)
			logEarray = vstack((logEarray,lE))

			lEps		= linspace(log(epsmin), log(epsmax), Esteps)
			logEpsArray	= vstack((logEpsArray,lEps))

			ta		= self.tau.opt_depth_array(self.zSource,exp(lE))[0]
			ta[ta > tauCut] = tauCut * ones(sum(ta > tauCut))
			tauArray	= vstack((tauArray,ta))
	    else:
		lg = linspace(g,gmax,gsteps)	# dim: epsGeV, gsteps
		logGamma = np.vstack((logGamma,lg))
		for j,lgi in enumerate(lg):
		    if not j:
			lEj	= linspace(log(2. * M_E_EV * 1e-12) + lgi, log(Emax), Esteps)
			lEpsj	= linspace(log(epsmin), log(epsmax) ,Esteps)
			taj	= self.tau.opt_depth_array(self.zSource,exp(lEj))[0]
		    else:
			lE 	= linspace(log(2. * M_E_EV * 1e-12) + lgi, log(Emax), Esteps)
			lEj	= vstack((lEj,lE))

			lEps	= linspace(log(epsmin), log(epsmax), Esteps)
			lEpsj	= vstack((lEps,lEpsj))

			ta	= self.tau.opt_depth_array(self.zSource,exp(lE))[0]
			ta[ta > tauCut] = tauCut * ones(sum(ta > tauCut))
			taj	= vstack((taj,ta))

		logEarray	= dstack((logEarray,lEj))	# dim: gsteps, Esteps, epsGeV
		logEpsArray	= dstack((logEpsArray,lEpsj))	# dim: gsteps, Esteps, epsGeV
		tauArray	= dstack((tauArray,taj))	# dim: gsteps, Esteps, epsGeV

	for i in range(Esteps):
	    if not i:
		logGammaArray = logGamma
	    else:
		logGammaArray = dstack((logGammaArray,logGamma))
	logGammaArray	= np.transpose(logGammaArray, axes = (1,2,0))
	logGamma	= logGamma.transpose()	# dim: gsteps, epsGeV

	EKernel		= self.intSpec(exp(logEarray),self.intSpecPar) * (exp(tauArray) - 1.) * exp(logEarray)\
			  * exp(-(exp(logEarray) / self.EmaxTeV) ** 10.)
	gammaKernel	= simps(EKernel,logEarray,axis = 1)


	x = epsGeV * 1e9 / (4. * exp(logEpsArray) * exp(2. * logGammaArray))

	x = ma.array(x, mask = x > 1)
	logEpsArray = ma.array(logEpsArray, mask = x > 1)

	EpsKernel = self._F_IC_T(x) * 4. * exp(2. * logGammaArray) * nphotCMBarray(exp(logEpsArray)) / exp(logEpsArray)
	EpsKernel *= exp(logEpsArray)

	EpsKernel = ma.array(EpsKernel, mask = x > 1)

	gammaKernel *= simps(EpsKernel,logEpsArray, axis = 1)
	gammaKernel /= exp(logGamma * 6.)
	gammaKernel *= exp(logGamma)

	result = simps(gammaKernel,logGamma, axis = 0)

	result *= M_E_EV / U_CMB * 9. / 64.

	return result,epsGeV

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
	#Emax = math.log(5e1)
	Emax = math.log(15.)
	if Emin > Emax:
	    gamma_kern_array[i] = 0.
	else:
	    log_E_array = np.linspace(Emin,Emax,E_steps)
	    E_inject_array = np.empty((len(log_E_array),))
	    for j,e in enumerate(log_E_array):
		E = np.exp(e)
		#E_inject_array[j] = E ** G * math.exp(-(E/Ebr) ** 10.)*( np.exp(tau(E)) - 1. )
		E_inject_array[j] = E ** G*( np.exp(tau(E)) - 1. )
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
