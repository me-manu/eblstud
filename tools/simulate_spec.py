"""
Module Containing tools for simulating spectra

History of changes:
Version 0.01
- Created 20th Feb 2014
"""

__version__ = 0.01
__author__ = "M. Meyer // manuel.meyer@fysik.su.se"

# - Imports ------------------------- #
import numpy as np
from numpy import meshgrid,exp,sqrt,pi,log10,linspace,isscalar,array,vstack,log,dstack,hstack,zeros
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.stats import poisson
import logging
from kapteyn import wcs
import pyfits
import my_fermi
import my_fermi.base.photons as fp
from os.path import join
# ----------------------------------- #

logging.basicConfig(level=logging.INFO)

# --- get r68 for Fermi --- #
def getPSFr68(Ereco, front = None, back = None):
	"""
	Get the r68 PSF confidence radius for front converted events for nominal incidence of Fermi-LAT

	Parameters
	----------
	Ereco:	n-dim array with reconstructed energies in MeV

	kwargs
	------
	front:	string, path to edisp fits file for front converted events. 
		If none, use P8 files strored in my_fermi/irfs/P8 (default = None)
	back:	string, path to edisp fits file for front converted events. 
		If none, use P8 files strored in my_fermi/irfs/P8 (default = None)

	Returns
	-------
	function pointer to r68 interpolation
	"""
	ct = np.ones(Ereco.shape[0])
	th = np.ones(Ereco.shape[0])


	if front == None or back == None:
	    path	= join(my_fermi.__path__[0],'irfs/P8')
	    front	= 'psf_P8SOURCE_V20R9P0_V0_front.fits'
	    back 	= 'psf_P8SOURCE_V20R9P0_V0_back.fits'

	psf	= fp.PSF(join(path , back),join(path , front))
	r68	= np.zeros(Ereco.shape[0])
	for i,eR in enumerate(Ereco):
	    r68[i]	= psf.calc_r68(0.68,eR,0,th[i]) * 180. / np.pi

	from scipy.interpolate import interp1d
	rInterp = interp1d(log(Ereco),r68)

	return lambda E: rInterp(log(E))

# --- Energy dispersion functions --- #
def edisp_Egauss(Etrue, Ereco, **kwargs):
    """
    Gaussion energy dispersion function

    Parameters
    ----------
    Etrue:	true energy, n-dim array
    Ereco:	reconstructed energy, m-dim array

    kwargs
    ------
    sE:		float, width of gaussian times energy, i.e. sigma = sE * Ereco

    Returns
    -------
    Energy dispersion as n x m dimensional array

    Note
    ----
    Formula: Edisp(Etrue,Ereco,sE) = exp(-0.5 * (Etrue - Ereco) **2 / (sE * Ereco) ** 2.) / sqrt(2. * pi) / sE / Ereco 
    """
    if isscalar(Ereco): Ereco = array([Ereco])
    if isscalar(Etrue): Etrue = array([Etrue])

    Err, Ett = meshgrid(Ereco,Etrue)

    try:
	result = exp(-0.5 * (Ett - Err) * (Ett - Err) / (kwargs['sE'] * kwargs['sE'] * Err * Err)) / sqrt(2. * pi) / kwargs['sE'] / Err
    except KeyError:
	logging.warning("*** edisp parameter kwarg do not contain 'sE' Keyword! Returning -1.")
	return -1
    return result

# --- The Class --------------------- #
class SimulateObs(object):
    """
    Class to simulate an observation given an energy dispersion, exposure, and source spectrum.

    Arguments
    ---------
    """

    def __init__(self,**kwargs):
	"""
	Initiate SimulateObs class.

	kwargs
	------
	src_spec:	source spectrum; function pointer, needs to be called like src_spec(energy, **src_par)
	src_par:	additional parameters for source spectrum

	exposure:	exposure; function pointer, needs to be called like exposure(energy, **exp_par)
	exp_par:	additional parameters for source spectrum

	edisp:		energy dispersion; function pointer, needs to be called like edisp(Ereco, Etrue, **edisp_par),
			where Ereco and Etrue are n,m - dim arrays and an n x m dim array is returned
	edisp_par:	additional paramerters for energy dispersion

	Returns
	-------
	Nothing.
	"""
# --- Set the defaults
	kwargs.setdefault('edisp',edisp_Egauss)
	kwargs.setdefault('edisp_par',{'sE':0.06})
# --------------------
	self.__dict__.update(kwargs)
	return

    def nPhotBin(self,ErecoBin, eTrueSteps = 0, eRecoSteps = 50, eSigma = 5.):
	"""
	compute expected number of photons in each energy bin

	Parameters
	----------
	ErecoBin:	(n+1)-dim array, containing the bin boundaries. Adjacent bins are assumed.

	kwargs
	------
	eTrueSteps:	int, number of integration steps for integration over energy dispersion. If zero, no energy dispersion assumed.
	eRecoSteps:	int, number of integration steps for integration over energy bin.
	eSigma:		float, number of sigma to integrate edisp over

	Returns
	-------
	n-dim array with expected number of photons in each energy bin.

	Notes:
	-----
	If energy bins change, energy cube is only recomputed if self.EbinBounds is deleted first.
	"""
	if (eSigma * self.edisp_par['sE'] >= 1.) and eTrueSteps:
	    logging.warning("*** eSigma * edisp_par['sE'] = {0:f} >= 1, has to be smaller than one, otherwise true energy is negative! Returning -1".format(eSigma * self.edisp_par['sE']))
	    return -1

	try:
	    self.EbinBounds
	except AttributeError:
	    self.EbinBounds = ErecoBin
	    self.dEbin	= zeros(ErecoBin.shape[0] - 1)
	    for i,e in enumerate(ErecoBin[:-1]):
		self.dEbin[i] = ErecoBin[i+1] - e
	
	    for iE, ErB in enumerate(ErecoBin[:-1]):
	# ---- build the reconstruced energy bin 2 x n array
		if not iE:
		    logEReco	= linspace(log(ErB),log(ErecoBin[iE + 1]),eRecoSteps)

		    if eTrueSteps:
			for jlE,lERi in enumerate(logEReco):
			    if not jlE:
				logETrue = linspace(log(exp(lERi) * \
				    (1. - eSigma * self.edisp_par['sE'])),log(exp(lERi) * (1. + eSigma * self.edisp_par['sE'])),eTrueSteps)
				eDisp = self.edisp(exp(logETrue),exp(lERi),**self.edisp_par)[:,0]

			    else:
				lET = linspace(log(exp(lERi) * \
				    (1. - eSigma * self.edisp_par['sE'])),log(exp(lERi) * (1. + eSigma * self.edisp_par['sE'])),eTrueSteps)
				logETrue = vstack((logETrue,lET))

				eD = self.edisp(exp(lET),exp(lERi),**self.edisp_par)[:,0]
				eDisp = vstack((eDisp,eD))

		else:
		    lER		= linspace(log(ErB),log(ErecoBin[iE + 1]),eRecoSteps)
		    logEReco	= vstack((logEReco,lER))	#dimensions : Bin, EReco in each bin

		    if eTrueSteps:
			for jlE,lERi in enumerate(lER):
			    if not jlE:
				lETi = linspace(log(exp(lERi) * \
				    (1. - eSigma * self.edisp_par['sE'])),log(exp(lERi) * (1. + eSigma * self.edisp_par['sE'])),eTrueSteps)

				eDi  = self.edisp(exp(lETi),exp(lERi),**self.edisp_par)[:,0]
			    else:
				lET = linspace(log(exp(lERi) * \
				    (1. - eSigma * self.edisp_par['sE'])),log(exp(lERi) * (1. + eSigma * self.edisp_par['sE'])),eTrueSteps)
				lETi= vstack((lETi,lET))

				eD  = self.edisp(exp(lET),exp(lERi),**self.edisp_par)[:,0]
				eDi = vstack((eDi,eD))

			logETrue	= dstack((logETrue,lETi))
			eDisp		= dstack((eDisp,eDi))
	    if eTrueSteps:
		self.eDisp	= eDisp.transpose()	# dimensions : Bin, ETrue, EReco in each bin
		self.logETrue	= logETrue.transpose()	# dimensions : Bin, ETrue, EReco in each bin
	    self.logEReco = logEReco
	    self.expAve	  = simps(self.exposure(exp(self.logEReco),self.exp_par) * exp(self.logEReco),logEReco, axis = 1) / simps(exp(self.logEReco),logEReco, axis = 1)
	# --- integrate over true energy and energy dispersion
	if eTrueSteps:
	    specEReco		= simps(self.eDisp * self.src_spec(exp(self.logETrue),self.src_par) * self.exposure(exp(self.logETrue),self.exp_par) * exp(self.logETrue),
					self.logETrue, axis = 1) 
	# --- integrate over reconstructed eneryg in each energy bin
	    self.nPhot		= simps(specEReco * exp(self.logEReco),self.logEReco,axis = 1)
	else:
	    self.nPhot		= simps(self.src_spec(exp(self.logEReco),self.src_par) * self.exposure(exp(self.logEReco),self.exp_par) * exp(self.logEReco),
					self.logEReco,axis = 1)
	# --- calculate average exposure in each energy bin

	return self.nPhot

    def simulateNphot(self, numSim = 1, flux = False):
	"""
	Simulate the number of events in each energy bin.
	The expected number of events is calculated by nPhotBin and stored in self.nPhot, the energy bin bounds are given in self.EbinBounds.
    
	kwargs
	------
	numSim:	integer, number of simulations
	flux:	boolean, if true, return simulated number of counts divided by exposure and bin width

	Returns:
	--------
	(numSim x EbinBounds.shape[0]) - dim array with poissonian random numbers.
	"""
	dummy = np.ones(numSim)			# dummy array for right shape	
	nn,dd = meshgrid(self.nPhot,dummy)	# nn: numsim x self.nPhot.shape dim array with self.nPhot in each row
	R = poisson.rvs(nn)			# do the random number generation

	if flux:
	    return R / self.dEbin / self.expAve
	else:
	    return R


    def setExposureFermi(self,ra=None,dec=None,exposure = None):
	"""
	Get the exposure from an exposure cube and interpolate it over energy.
	Sets self.exposure function.

	kwargs
	------
	ra:	float, right ascension of the source, in degrees. If None, use self.ra
	dec:	float, declination of the source, in degrees. If None, use self.dec
	exposure:	fits file with exposure cube (output of Fermi LAT science tools tool gtexpcube2).
		    	If not specified, self.expCubeFits is used.

	Returns
	-------
	Function pointer to exposure.
	"""

	if not exposure == None: self.expCubeFits = exposure
	if not ra == None: self.ra = ra
	if not dec == None: self.dec = dec

	expos		= pyfits.open(self.expCubeFits)
	e_h, e_d	= expos[0].header, expos[0].data	# data format is Energy, b, l
	e_EMeV		= expos[1].data.field(0)
	proj_e		= wcs.Projection(e_h)

	
	if e_h['CTYPE2'].find('GLAT') >= 0. and e_h['CTYPE1'].find('GLON') >= 0.:
	    tran		= wcs.Transformation("EQ,fk5,J2000.0", "GAL")
	    self.l,self.b	= tran.transform((self.ra,self.dec))

	    pix_e = proj_e.topixel((self.l,self.b,0.))	# first pixel: l, second pixel b, third pixel E, sequence reveresed compared to data
	    pix_e = array(pix_e)
	if e_h['CTYPE2'].find('DEC') >= 0. and e_h['CTYPE1'].find('RA') >= 0.:
	    pix_e = proj_e.topixel((self.ra,self.dec,0.))	# first pixel: ra, second pixel dec, third pixel E, sequence reveresed compared to data
	    pix_e = array(pix_e)


	logExp	= interp1d(log(e_EMeV),log(e_d[:,pix_e[1],pix_e[0]]))	# interpolate over the energy 

	#self.exposure	= lambda EMeV,**pars: exp(logExp(log(EMeV)))
	self.exposure	= lambda EMeV,pars: exp(logExp(log(EMeV)))
	self.exp_par	= {}
	self.EExpMeV	= e_EMeV

	return

    def setGalDiffFermi(self,ra=None,dec=None,galDiff = None):
	"""
	Get the galactic diffuse emission from a fits file and interpolate it over energy.
	Sets self.galDiff function.

	kwargs
	------
	ra:	float, right ascension of the source, in degrees. If None, use self.ra
	dec:	float, declination of the source, in degrees. If None, use self.dec
	galDiff:	fits file with galactic diffuse emission model.
		    	If not specified, self.galDiffFits is used.

	Returns
	-------
	Function pointer to exposure.
	"""

	if not galDiff == None: self.galDiffFits= galDiff
	if not ra == None: self.ra = ra
	if not dec == None: self.dec = dec

	gal		= pyfits.open(self.galDiffFits)
	g_h, g_d	= gal[0].header, gal[0].data	# data format is Energy, b, l
	g_EMeV		= gal[1].data.field(0)
	proj_g		= wcs.Projection(g_h)	# get the projection pixel to world, they are in galactic coordinates.

	tran		= wcs.Transformation("EQ,fk5,J2000.0", "GAL")
	self.l,self.b	= tran.transform((self.ra,self.dec))

	pix_g = proj_g.topixel((self.l,self.b,0.))	# first pixel: l, second pixel b, third pixel E, sequence reveresed compared to data
	pix_g = array(pix_g)

	logGalFlux	= interp1d(log(g_EMeV),log(g_d[:,pix_g[1],pix_g[0]]))	# interpolate over the energy 

	self.galDiff	= lambda EMeV: exp(logGalFlux(log(EMeV)))
	self.EGalMeV	= g_EMeV

	return

    def setIsoDiffFermi(self,isoDiff = None):
	"""
	Get the isotropic diffuse emission from a text file and interpolate it over energy.
	Sets self.isoDiff function.

	kwargs
	------
	isoDiff:	text file with isotropic diffuse emission model.
		    	If not specified, self.isoDifftxt is used.

	Returns
	-------
	Function pointer to exposure.
	"""

	if not isoDiff == None: self.isoDifftxt = isoDiff

	iso		= np.loadtxt(self.isoDifftxt)

	logIso		= interp1d(log(iso[:,0]),log(iso[:,1])) 

	self.isoDiff	= lambda EMeV: exp(logIso(log(EMeV)))
	self.EisoMeV	= iso[:,0]

	return
	
    def setEdispFermi(self, Ereco, eTrueSteps = 300, front = None, back = None):
	"""
	Set energy dispersion to Fermi's energy dispersion, assuming theta = 0 and front conversion,
	using 2d interpolation.

	Parameters
	----------
	Ereco:	n-dim array with reconstructed energies in MeV

	kwargs
	------
	eTrueSteps:	True energy steps
	front:	string, path to edisp fits file for front converted events. 
		If none, use P8 files strored in my_fermi/irfs/P8 (default = None)
	back:	string, path to edisp fits file for front converted events. 
		If none, use P8 files strored in my_fermi/irfs/P8 (default = None)

	Returns
	-------
	"""
	ct = np.ones(Ereco.shape[0])
	th = np.ones(Ereco.shape[0])


	if front == None or back == None:
	    #path	= join(my_fermi.__path__[0],'irfs/P8')
	    #front	= 'edisp_P8SOURCE_V20R9P0_V0_front.fits'
	    #back 	= 'edisp_P8SOURCE_V20R9P0_V0_back.fits'
	    path	= join(my_fermi.__path__[0],'irfs/P7REP')
	    front	= 'edisp_P7REP_CLEAN_V15_front.fits'
	    back 	= 'edisp_P7REP_CLEAN_V15_back.fits'

	edisp	= fp.EDISP(join(path , back),join(path , front))
	Ed	= np.zeros((eTrueSteps,Ereco.shape[0]))
	Etrue = 10.**np.linspace(np.log10(Ereco[0]) - 2., np.log10(Ereco[-1]) + 2.,eTrueSteps)
	for j,eR in enumerate(Ereco):
	    #N = edisp.p_E_disp_x(5.,eR,ct[j],th[j])
	    N = 1.
	    for i,eT in enumerate(Etrue):
		x 	= (eR - eT) / (eT * edisp.scaling_factor(eT,th[j],ct[j]))
		Ed[i,j]	= edisp.E_disp(x,eR,th[j],ct[j]) / N
		if Ed[i,j] == 0.:
		    Ed[i,j] = 1e-40

	from scipy.interpolate import RectBivariateSpline as RBSpline

	self.scaling_factor = edisp.scaling_factor
	self.edispInterp = RBSpline(log(Etrue),log(Ereco),log(Ed),kx = 2, ky = 2)
	self.edisp = lambda Etrue,Ereco,**par: exp(self.edispInterp(log(Etrue),log(Ereco)))

	return

    def plot_edisp(self,Emin,Emax):
	"""
	Convenience function to plot energy dispersion.

	Parameters:
	-----------
	Emin:	float, minimum energy 
	Emax:	float, maximum energy 

	Returns:
	--------
	Nothing.
	"""
	import matplotlib.pyplot as plt

	Etrue = 10. ** linspace(log10(Emin),log10(Emax),100)
	Ereco = 10. ** linspace(log10(Emin),log10(Emax),200)

	logEdisp = log10(self.edisp(Etrue,Ereco,**self.edisp_par))

	im = plt.imshow(logEdisp,
	    vmin = -10,vmax = 0.,
	    extent = (log10(Ereco[0]),log10(Ereco[-1]),log10(Etrue[0]),log10(Etrue[-1])),
	    cmap= plt.cm.jet,origin = 'low'

	    )
	cbar = plt.colorbar(im)
	cbar.ax.set_ylabel('Energy dispersion (1 / Energy)')
	plt.xlabel('$\log_{10}E_\mathrm{true}$')
	plt.ylabel('$\log_{10}E_\mathrm{reco}$')
	plt.show()

	return
