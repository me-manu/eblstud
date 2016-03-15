# --- Imports -----------------------------------#
from eblstud.misc.constants import M_E_EV,R_E_CM
import numpy as np
# -----------------------------------------------#
"""
Class to calculate absorption in BLR using Justin's fitting formula.
"""

__version__ = 1.

class OptDepth_BLR(object):
    """
    Class to calculate the optical depth in the broad-line region (BLR) 
    using model by J. Finke (2015)

    Attributes
    ----------
    Elines:	Line energies in the BLR, in eV
    Nlines: 	log10 column densities of emission lines in the BLR, in  cm^-2
    z:		Redshift of the source
    A:		Fitting constants for polynomial fit to absorption
    """
    def __init__(self, **kwargs):

	kwargs.setdefault('z',0.)
	kwargs.setdefault('Elines',np.array([13.6,10.2,8.0,24.6,21.2]))
	kwargs.setdefault('Nlines',np.array([24.,24.,24.,24.,24.]))
	kwargs.setdefault('A',np.array([-0.34870519,0.50134530,
					-0.62043951,-0.022359406,
					0.059176392,0.0099356923]))
        self.__dict__.update(kwargs)

	return 

    def __calctau(self,EGeV,**kwargs):
	"""
	Helper function to compute optical depth
	"""

	ElEl, EE = np.meshgrid(kwargs['Elines'], EGeV)
        x = EE * 1.e9 * (1. + self.z)*ElEl / M_E_EV / M_E_EV #1e9 added since E is in GeV
        m = x > 1.
        n = x > 10.
        y = np.zeros(x.shape)
        y[m] = np.log10(x[m]-1.)
	y.reshape(x.shape)

	exponent = np.zeros(y.shape)
	for i,a in enumerate(self.A):
	    exponent[m] += a * np.power(y[m],i)
	exponent.reshape(y.shape)

	tau = np.zeros(y.shape)
        tau[m] = np.power(10., exponent[m])
	tau.reshape(y.shape)
        tau[n] = ((2.*x[n]*( np.log(4.*x[n]) - 2. ) + \
		np.log(4.*x[n])*( np.log(4*x[n]) - 2. ) - (np.pi*np.pi - 9.)/3.)) \
		/ (x[n] * x[n])
	tau.reshape(y.shape)
	tau *= np.pi * np.power(10.,kwargs['Nlines']) * R_E_CM * R_E_CM
        return tau
		
    def __call__(self,EGeV,**kwargs):
	"""
	Compute optical depth for one combination of line energy and column density. 
	
	Parameters
	----------
	EGeV:	m-dim array, photon energies, in GeV

	kwargs
	------
	Elines: n-dim array, line energy, in eV
	Nlines: n-dim array,  log 10 column density, in cm^-2
	z:	float, redshift

	Returns
	-------
	Optical depth as (n x m)-dim array
	"""

	if np.isscalar(EGeV):
	    E = np.array([EGeV])
	kwargs.setdefault('Elines',self.Elines)
	kwargs.setdefault('Nlines',self.Nlines)
	if np.isscalar(kwargs['Elines']):
	    kwargs['Elines'] = np.array(kwargs['Elines'])
	if np.isscalar(kwargs['Nlines']):
	    kwargs['Nlines'] = np.array(kwargs['Nlines'])
	if not len(kwargs['Nlines']) == len(kwargs['Elines']):
	    raise ValueError("Nlines and Elines do not match. Shapes: {0}, {1}".format(
		kwargs['Nlines'].shape, kwargs['Elines'].shape
		))
    	return self.__calctau(EGeV,**kwargs).T
