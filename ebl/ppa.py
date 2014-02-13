"""
Functions needed to calculate pair production anomaly
"""
import numpy as np
import eblstud.ebl.tau_from_model as TAU
import eblstud.tools.lsq_fit as lf
import matplotlib.pyplot as plt
import vhe_spectra
import pickle
import sys,os
from matplotlib import pyplot, mpl
import eblstud.stats.ks as ks
import eblstud.stats.anderson_darling as ad
from scipy.stats import t as tstat
from scipy.stats import norm as norm_dist
from scipy.special import ndtri
import logging
from math import ceil
import subprocess
import eblstud.tools.iminuit_fit as mf

def plot_extra_legend() :
    from matplotlib.lines import Line2D
    c = '.5'
    l1 = Line2D([], [], color=c, marker='o', ms=(0.05/0.536 * 5.)**1.5 + 4.5, markeredgewidth=0., mec='White', ls='')
    l2 = Line2D([], [], color=c, marker='o', ms=(0.2/0.536 * 5.)**1.5 + 4.5, markeredgewidth=0., mec='White', ls='')
    l3 = Line2D([], [], color=c, marker='o', ms=(0.5/0.536 * 5.)**1.5 + 4.5, markeredgewidth=0., mec='White', ls='')
    l4 = Line2D([], [], color='black', marker='', ls='-',lw = 3)
    l5 = Line2D([], [], color=c, marker='*', ms=14, markeredgewidth=0., mec='White', ls='')
    plt.legend([l1, l2, l3, l5,l4],
    ['$z$ = 0.05','$z$ = 0.2','$z$ = 0.5', 'Mean values','Smoothed average'],
    prop={'size': 'medium'}, numpoints=1, loc=4, frameon=False)
def plot_extra_legend2(col = '0.8', lcol = 'red') :
    from matplotlib.lines import Line2D
    c = col
    l1 = Line2D([], [], color=c, marker='o', ms=(0.05/0.536 * 5.)**1.5 + 4.5,  ls='')
    l2 = Line2D([], [], color=c, marker='o', ms=(0.2/0.536 * 5.)**1.5 + 4.5,  ls='')
    l3 = Line2D([], [], color=c, marker='o', ms=(0.5/0.536 * 5.)**1.5 + 4.5,  ls='')
    l4 = Line2D([], [], color=lcol, marker='', ls='-',lw = 3)
    plt.legend([l1, l2, l3, l4],
    ['$z$ = 0.05','$z$ = 0.2','$z$ = 0.5','Smoothed average'],
    prop={'size': 'x-large'}, numpoints=1, loc=4, frameon=True)


RatioCalc       =  lambda fint, fext: (fint - fext) / (fint + fext) # fint: intrinsic measured flux, 
                                                                    # fext: extrapolated flux
ResidualCalc    =  lambda fint, fext,serr: (fint - fext) / serr     # fint: intrinsic measured flux, 
								    # fext: extrapolated flux
								    # serr: stat. uncertainty 

class PPA(object):
    """
    Class to calculate pair production anomaly from VHE gamma-ray spectra

    Attributes
    ----------
    tau:	n-dim-array with all optical depth values of all data points in the sample
    tau_ratio:	m-dim-array with all optical depth values for KS test
    energy:	n-dim-array with all energy values of all data points in the sample
    redshift:	n-dim-array with all redshift values of all data points in the sample
    residual:	n-dim-array with all residual values of all data points in the sample determined from a chi2 fit
    """

    def __init__(self):
	"""
	Initialize PPA class to calculate significance of pair production anomaly
	"""

	self.tau	= np.array([])
	self.tau_ratio	= np.array([])
	self.energy	= np.array([])
	self.energy_ratio	= np.array([])
	self.redshift	= np.array([])
	self.redshift_ratio	= np.array([])
	self.ratio	= np.array([])
	self.residual	= np.array([])

	return

    def clear(self):
	"""
	clear all PPA arrays
	"""
	self.tau	= np.array([])
	self.tau_ratio	= np.array([])
	self.energy	= np.array([])
	self.energy_ratio	= np.array([])
	self.redshift	= np.array([])
	self.redshift_ratio	= np.array([])
	self.ratio	= np.array([])
	self.residual	= np.array([])
	return

# ------------------------------------------------------------------------------------------#
# --- Test the pair priduction anomaly -----------------------------------------------------#
# ------------------------------------------------------------------------------------------#
    def ppa(self,meas,eblmodel, TauThinLim = 1., TauThickLim = 2., TauScale = 1., EScale = 1., RemovePoints = 0, observed_spec = False,  ks_plot = 'None',
    		save_fit_stat = False,file_name = 'None', use_pure_borders = True, w_ALPs = 'None'):
	"""
	function to calculate pair production anomaly with KS- and t-Test
	using the iminuit package

	Parameters
	----------
	meas:		list of dictionaries with VHE spectral measurements
	eblmodel:	ebl model, possibilities are: kneiske, franceschini or dominguez
	TauThinLim:	lower limit for tau distribution of opt. thin regime (defualt: 1.)
	TauThickLim: 	limit on optical depth for opt. thick regime (defualt: 2.)
	TauScale:	Additional scale on the optical depth. default: 1
	EScale:		Additional scale on the measured energy. default: 1
	RemovePoints: 	index up to which spectral points are taken into account. default: all points are included
	observed_spec:	bool, if true, no absortion correction is applied to spectral points
	ks_plot:	String (needs to end on .pdf), if not None, KS plots will be produced and saved to ks_plot
	save_fit_stat:	bool, if True, fit statistics are saved in np.arrays
	file_name:	EBL file name
	use_pure_borders:	If true, use EBL model for tau borders even if file_name is supplied
	w_ALPs:		if not None, yaml file to calculate deabs. with ALPs

	Returns
	-------
	pKS:		p-value of KS test
	pT:		p-value of T-test


	Notes
	-----
	"""

# --- Start to loop over all spectra ----------------------------------------------------#

	ebl = TAU.OptDepth()
	ebl.readfile(model = eblmodel, file_name = file_name)

	ebl_fm = TAU.OptDepth(model = eblmodel)	# pure model EBL

	self.TauThinLim		= TauThinLim
	self.TauThickLim	= TauThickLim
	self.eblmodel		= eblmodel
	self.TauScale		= TauScale

	if not w_ALPs == 'None':
	    import PhotALPsConv.calc_conversion as CC
	    from eblstud.astro.coord import RaHMS2float,DecDMS2float
	    cc = CC.Calc_Conv(config = w_ALPs)

	#meas = sorted(meas, key=lambda m: m['z'])	# sort by redshift
	if save_fit_stat:
	    id_array	= []
	    chi2_array	= []
	    dof_array	= []
	    pval_array	= []
	    pfin_array	= []
	    E_range	= []
	    fit_func_array = []

	for m in meas:
	    if m['z_uncertain']:
		continue
		#if not m['object'] == 'PKS1424+240':
		    #continue
	    if m['id'].find("ave") >= 0:	# no average spectra
		continue
# only combined spectrum of H1426 
	    if m['id'] == 'H1426_HEGRA_2002' \
		or m['id'] == 'H1426_HEGRA_2000reanal' \
		or m['id'] == 'RGBJ0710_VERITAS_2010he'\
		or m['id'] == 'RXSJ1010_HESS_2011':
		continue
	    if m['z'] < 0.01:
		continue

# --- Deabsorb spectra and sort them into optical thin and thick samples -----------------#
	    if RemovePoints:
		x = np.array(m['data']['energy'])[:RemovePoints]*EScale
		y = np.array(m['data']['flux'][:RemovePoints])
		s = np.array(m['data']['f_err_sym'][:RemovePoints])
	    else:
		x = np.array(m['data']['energy'])*EScale
		y = np.array(m['data']['flux'])
		s = np.array(m['data']['f_err_sym'])
	    if not TauScale == 0.:
		if use_pure_borders:
		    t	= ebl_fm.opt_depth_array(m['z'],x)[0] * TauScale
		    tf	= ebl.opt_depth_array(m['z'],x)[0] * TauScale
		else:
		    t = ebl.opt_depth_array(m['z'],x)[0] * TauScale
		    tf = t
	    else:
		t = ebl.opt_depth_array(m['z'],x)[0] * TauScale

	    if not observed_spec:
		if w_ALPs == 'None':
		    ydeabs = y * np.exp(tf)
		    sdeabs = s * np.exp(tf)
		else:
		    cc.kwargs['ebl']	= eblmodel
		    cc.kwargs['ebl_norm']	= TauScale
		    cc.kwargs['ra']	= RaHMS2float(m['ra'])
		    cc.kwargs['dec']	= DecDMS2float(m['dec'])
		    cc.kwargs['z']	= m['z']
		    cc.update_params_all(**cc.kwargs)
		    Pt,Pu,Pa = cc.calc_conversion(x * 1e3, new_angles = True)
		    ydeabs = y / (Pt + Pu)
		    sdeabs = s / (Pt + Pu)
	    else:
		ydeabs = y
		sdeabs = s

	    if np.any(t >= TauThinLim):
		logging.debug("{0:20}{1:10.3f}  {2:20}{3:.2f} - {4:.2f}  TeV\t{5:.2f} {6} {7} {8}".format(m['object'],m['z'],m['instrument'],np.min(x),np.max(x),np.max(t),t[t<1].shape[0],t[(t>=1) & (t < 2)].shape[0],t[t >= 2].shape[0]))

    # to get the thin indices, x has to be sorted in ascending order
	    for i,e in enumerate(x):
		if not e == np.sort(x)[i]:
		    raise ValueError('Spectrum must be sorted in ascending order in energy!')
	    fit		= np.where(t < TauThinLim)
	    thin	= np.where(t[np.where(t < TauThickLim)] >= TauThinLim)
	    thick	= np.where(t >= TauThickLim)

# --- Start the fitting ------------------------------------------------------------------#
# --- try PL, LP as fit functions  -------------------------------------------------------#

	    #minf    = mf.FitMinuit()
	    fit_func= mf.MinuitFitPL,mf.MinuitFitLP
	    func	= mf.pl,mf.lp

	    if len(x) < 3:
		continue

	    for i in range(len(fit_func)):

		#fit_stat,pfinal,fit_err,covar = fit_func[i](
		fit_stat,pfinal,fit_err = fit_func[i](
		    x,
		    ydeabs*x**2.,
		    1.2*sdeabs*x**2.,
		    #full_output = True
		    full_output = False
		    )
		#pfinal[1] -= 2.
		try:
		    pfinal['Index'] -= 2.
		except KeyError:
		    pfinal['alpha'] -= 2.
		fit_stat,pfinal,fit_err = fit_func[i](
		    x,
		    ydeabs,
		    sdeabs,
		    pinit = pfinal, 
		   # full_output = True
		    full_output = False
		    )
	    # Consider different norm for logpar (not 1 TeV!)
		func_ext = func[i]
		if fit_stat[2] > 0.05 or len(x) < 4:
		    break
		if fit_stat[2] < 0.05:
		    logging.warning("Bad Fit detected! Chi^2 = {0}, p-value = {1}, in spec id {2}".format(fit_stat[0]/fit_stat[1],fit_stat[2],m['id']))

	    if save_fit_stat and np.any(t >= TauThickLim):		# save fit statistics
		id_array.append(m['object'] + " " + m['instrument'])
		chi2_array.append(fit_stat[0])
		dof_array.append(fit_stat[1])
		pval_array.append(fit_stat[2])
		pfin_array.append(pfinal)
		E_range.append([x[0],x[-1]])
		fit_func_array.append(i)

	    self.tau		= np.concatenate( (self.tau,t) )
	    self.energy		= np.concatenate( (self.energy,x) )
	    self.redshift	= np.concatenate( (self.redshift,m['z']*np.ones(x.shape[0])) )
	    self.residual	= np.concatenate( (self.residual, ResidualCalc( ydeabs,func_ext(pfinal,x),sdeabs )) )

	    #print m['id'], len(x[fit])
	    cont = False
	    if len(x[fit]) == 2:
	    # Exclude all spectra with 2 points or less in x[fit]
#           continue
		fit_stat    = 0.,2.,1.
		fit_err     = {"Scale": 0., "Index": 0., "Prefactor": 0.}
		func_ext    = func[0]
		pfinal	    = {}
		pfinal["Scale"] = x[np.argmax(ydeabs / sdeabs)]
		pfinal["Index"] = (np.log(ydeabs[fit][1]) - np.log(ydeabs[fit][0])) / \
		(np.log(x[fit][1]/pfinal["Scale"]) - np.log(x[fit][0]/pfinal["Scale"]))
		pfinal["Prefactor"]= np.exp(np.log(ydeabs[fit][0]) - pfinal["Index"] * np.log(x[fit][0]/pfinal["Scale"]))
	    elif len(x[fit]) > 2:
		for i in range(len(fit_func)):

		    #fit_stat,pfinal,fit_err,covar = fit_func[i](
		    fit_stat,pfinal,fit_err = fit_func[i](
			x[fit],
			ydeabs[fit]*x[fit]**2.,
			1.2*sdeabs[fit]*x[fit]**2.,
			#full_output = True
			full_output = False
			)
		    try:
			pfinal["Index"] -= 2.
		    except KeyError:
			pfinal["alpha"] -= 2.
		    #fit_stat,pfinal,fit_err,covar = fit_func[i](
		    fit_stat,pfinal,fit_err = fit_func[i](
			x[fit],
			ydeabs[fit],
			sdeabs[fit],
			pinit = pfinal, 
			#full_output = True
			full_output = False
			)
		# Consider different norm for logpar (not 1 TeV!)
		    func_ext = func[i]
		    if fit_stat[2] > 0.05:
			break
		    if fit_stat[2] < 0.05:
			logging.warning("Bad Fit detected! Chi^2 = {0}, p-value = {1}, in spec id {2}".format(fit_stat[0]/fit_stat[1],fit_stat[2],m['id']))
			if i or len(x[fit]) < 4:
			    logging.warning('Fit stays bad with log par / or not enough data points, continuing with next spectrum')
			    cont = True
			    break
	    else:
		continue
	    if cont:	# fit is bad: continue with next spectrum
		continue
	    self.tau_ratio	= np.concatenate( (self.tau_ratio,t) )
	    self.ratio		= np.concatenate( (self.ratio, RatioCalc( ydeabs,func_ext(pfinal,x) ) ))
	    self.redshift_ratio	= np.concatenate( (self.redshift_ratio,m['z']*np.ones(x.shape[0])) )
	    self.energy_ratio	= np.concatenate( (self.energy_ratio, x))



	mt_thin  = (self.tau >= TauThinLim) & (self.tau < TauThickLim)	# mask for opt thin distr. for t test
	mt_thick = self.tau >= TauThickLim			 	# mask for opt thick distr. for t test

	mtr_thin  = (self.tau_ratio >= TauThinLim) & (self.tau_ratio < TauThickLim)	# mask for opt thin distr. for KS test
	mtr_thick = self.tau_ratio >= TauThickLim			 	# mask for opt thick distr. for KS test

	self.pKS	= ks.KStest(self.ratio[mtr_thin],self.ratio[mtr_thick])[1]		# p-value of KS-test

	mv  = np.mean(self.residual[mt_thick]), np.std(self.residual[mt_thick], ddof = 1) # mean and std of residual distribution
# use weighted quantites
	wmean	= np.sum(self.residual[mt_thick] * self.tau[mt_thick]) / np.sum(self.tau[mt_thick])
	wstd	= np.sqrt(np.sum(self.tau[mt_thick] * (self.residual[mt_thick] - wmean) ** 2.) / (np.sum(self.tau[mt_thick]) - 1.))
#	mv	= (wmean, wstd)


	pAD = ad.And_Darl_Stat(self.residual[mt_thick])			# p-value of Anderson Darling Test, that residual[mt_thick] follows normal distr.
	if pAD < 0.05:
	    logging.warning("probability of Anderson Darling test is small: {0}".format(pAD))

	T   = mv[0] / mv[1] * np.sqrt(np.sum(mt_thick))
	self.pT  = tstat.cdf(T, np.sum(mt_thick) - 1)			# p-value of T-test that distribution follows Gaussian with zero mean

	if save_fit_stat:		# save fit statistics
	    self.fit_func_array	= np.array(fit_func_array)
	    self.id_array	= np.array(id_array)
	    self.chi2_array	= np.array(chi2_array)
	    self.dof_array	= np.array(dof_array)
	    self.pval_array	= np.array(pval_array)
	    self.pfin_array	= np.array(pfin_array)
	    self.E_range	= np.array(E_range)

	if not ks_plot == 'None':
# DEBUG:
	#if not ks_plot == 'None' and self.pKS < 1e-6:
	    ks.plot_KStest1(self.ratio[mtr_thin],self.ratio[mtr_thick], filename = ks_plot, xlabel = 'Ratio')
	return self.pKS, self.pT

# ------------------------------------------------------------------------------------------#
# --- Mean vs Tau Plot and Histograms ------------------------------------------------------#
# ------------------------------------------------------------------------------------------#
    def plot_residual_vs_tau(self,LOESS = True, filename=None, fmt = 'pdf'):
	"""
	function to plot residuals vs tau plance together with histograms and LOESS smooth average curve

	Parameters:
	-----------
	LOESS:		bool, if True LOESS regression is plotted (R needs to be installed!)
	filename:	string, filename for output plot
	format:		string, format of output plot (pdf, eps, png)

	Returns
	-------
	Nothing
	"""

	import matplotlib
	import matplotlib.pyplot as plt

	fig = plt.figure(num = 1, figsize=(12,9))
	gs = matplotlib.gridspec.GridSpec(3,13)

	cp = plt.cm.copper

	cuts	= [self.tau < 1., (self.tau >= 1.) & (self.tau < 2.), self.tau > 2.]

# --- PLOT the Histograms ------------------------------------------------------#
	labels	= [r"$\tau\,<\,1$", r"$1\,\leqslant\,\tau\,<\,2$", r"$2\,\leqslant\,\tau$"]
	for i,c in enumerate(cuts):
	    if np.sum(c) > 20:
		bins = 10
	    else:
		bins = 5
	
	    ax = plt.subplot(gs[0,i*4:i*4+4])
	    n,bins,patches = plt.hist(self.residual[c], normed = True,label = labels[i], bins = bins, edgecolor = cp(1. - i/5.), facecolor = cp(1.-i/5.))	# plot the histogram

	    mv  = np.mean(self.residual[c]), np.std(self.residual[c], ddof = 1) # mean and std of residual distribution
	    sm, sv = np.abs(mv[0] / np.sqrt(np.sum(c))), mv[1] / np.sqrt(2. * (np.sum(c) - 1.))	# error on mean and standard deviation
	    plt.plot(np.linspace(bins[0],bins[-1],100),				# plot the Gaussion
	    	norm_dist.pdf(np.linspace(bins[0],bins[-1],100),loc = mv[0], scale = mv[1]),
		color='black',
		lw=2
		)
	    string = "$\\bar\\chi = {0:.2f}\pm{1:.2f}$\n$\\sigma_\\chi = {2:.2f}\pm{3:.2f}$".format(mv[0],sm, mv[1],sv)
	    plt.annotate(string, xy = (0.05,0.7), xycoords = 'axes fraction', size = 11)

	    if i:
		for tick in ax.yaxis.get_major_ticks():
		    tick.label1On = False
		xlabels = ax.get_xticklabels()
		xlabels[0].set_visible(False)
	    plt.axis([-3.,3.,0.,1.0])
	    plt.legend(loc = 1, frameon = True,prop = {'size':'small'})
	    for label in ax.xaxis.get_ticklabels():
		label.set_fontsize('small')
	    for label in ax.yaxis.get_ticklabels():
		label.set_fontsize('small')

	    if not i:
		plt.ylabel(r"Density",size = 'medium')
		plt.xlabel(r"Residuals $\chi$",size = 'medium', x = 1.5)


	plt.subplots_adjust(hspace = 0.3, wspace = 0.0)

# --- LOESS regression ---------------------------------------------------------------------#
	if LOESS:
	    fi = open('RvsT.dat','w')
	    for i,R in enumerate(self.residual):
		fi.write('{0}\t{1}\n'.format(self.tau[i], R))
	    fi.close()

	    try:
		if os.environ["HOST"] == "uh2ulastro15":
		    subprocess.call(['R','-q','-f','script.r'])
		elif os.environ["HOST"] == "crf-wgs01" or os.environ["HOST"] == "astro-wgs01":
		    subprocess.call(['/nfs/astrop/d1/software/R/bin/R','-q','-f','script.r'])
		else:
		    logging.warning("R is not installed! Exit.")
	    except KeyError:
		try:
		    subprocess.call(['R','-q','-f','script.r'])
		except:
		    logging.warning("R is not installed! Exit.")
	    fi = open('loess.info')
	    for l in fi.readlines():
		if l.find('Equivalent') >= 0:
		    eff_num_params = float(l.split(':')[1])
		if l.find('Residual') >= 0:
		    res_err = float(l.split(':')[1])
		if l.find('Trace') >= 0:
		    trace_smooth = float(l.split(':')[1])
	    loess = np.loadtxt('loess.dat').transpose()

# --- Residuals vs Tau Plot ----------------------------------------------------------------#

	ax = plt.subplot(gs[1:,:-1])
	cp = plt.cm.RdBu

	plt.axvline(self.TauThinLim, linestyle ='--', color = 'black', lw = 1.5)
	plt.axvline(self.TauThickLim, linestyle ='-', color = 'black', lw = 1.5)
	plt.axhline(0., linestyle ='-', color = '0.5', lw = 2.)

	xval = (self.TauThinLim/2.,(self.TauThickLim + self.TauThinLim)/2.,(2. + ceil(np.max(self.tau[cuts[-1]])))/2.)
	xerr = (0.5,0.5,(ceil(np.max(self.tau[cuts[-1]])) - 2.)/2.)

	for i,c in enumerate(cuts):
	    for j,E in enumerate(self.energy[c]):
		plt.plot((self.tau[c])[j],(self.residual[c])[j],
		    linestyle = 'None',
		    marker = 'o',
		    ms = ((self.redshift[c])[j]/0.536 * 5.)**1.5 + 4.5,
		    color = cp( ( np.log10(E/0.1)/np.log10(10./0.1)) ),
		    mec = '0.'
		    )
	    mv  = np.mean(self.residual[c]), np.std(self.residual[c], ddof = 1) # mean and std of residual distribution
	    sm, sv = np.abs(mv[0] / np.sqrt(np.sum(c))), mv[1] / np.sqrt(2. * (np.sum(c) - 1.))	# error on mean and standard deviation

	    plt.errorbar((xval[i]),mv[0],
		yerr = (sm),
		xerr = (xerr[i]),
		linestyle = "none",
		marker = '*',
		color = plt.cm.copper(1. -  i/5.),
		#mec = plt.cm.copper(1. - i/5.),
		ms = 14.,
		lw = 2.,
		)
	if LOESS:
	    plt.plot(np.sort(loess[0]),(loess[1])[np.argsort(self.tau)],lw = 3, color = 'black',label = 'Loess Regression')

# --- Set EBL String:
	ebl = ['Kneiske & Dole (2010)', 'Franceschini $et\,al.$ (2008)', 'Dominguez $et\,al.$ (2011)',\
	    'Inoue $et\,al.$ (2012)']
	if self.eblmodel== 'kneiske':
	    string = ebl[0]
	elif self.eblmodel == 'franceschini':
	    string = ebl[1]
	elif self.eblmodel == 'dominguez':
	    string = ebl[2]
	elif self.eblmodel == 'inoue':
	    string = ebl[3]
	plt.annotate('EBL: {0}, $\\alpha = {1}$\n$p_t = {2:.2f}$'.format(string,self.TauScale,ndtri(self.pT)),
	    xy = (0.95,0.95), xycoords = 'axes fraction', ha = 'right', va = 'top', size = 'medium')
	plt.xlabel(r'Optical depth $\tau$', size = 'large')
	plt.ylabel('Residuals $\chi$', size = 'large')

	ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
	ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
	ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.2))
	ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.2))
	plot_extra_legend()

	plt.axis([0.,ceil(np.max(self.tau)),-3.,4.])
	ax1 = fig.add_axes([0.86, 0.1, 0.03, 0.51])

	# Set the colormap and norm to correspond to the data for which
	# the colorbar will be used.
	#norm = mpl.colors.Normalize(vmin=0, vmax=23.1)
	norm = mpl.colors.LogNorm(vmin=0.1, vmax=10.0)

	# ColorbarBase derives from ScalarMappable and puts a colorbar
	# in a specified axes, so it has everything needed for a
	# standalone colorbar.  There are many more kwargs, but the
	# following gives a basic continuous colorbar with ticks
	# and labels.
	cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cp,
	    norm=norm,
	    orientation='vertical',
	    ticks=np.hstack((np.linspace(0.1,1.,10),np.linspace(1.,10.,10)))#[10,1.,0.1]
	    )
	cb1.ax.set_yticklabels(['0.1','','','','','','','','','', '1','','','','','','','','','10'])
	cb1.set_label('Energy (TeV)',size='large')

	if filename == None:
	    plt.savefig('../plots/scatter_resid_vs_tau_{0}_{1}.{2}'.format(self.eblmodel,str(self.TauScale).replace(".","p"),fmt), format = fmt)
	else:
	    plt.savefig(filename, format = fmt)

	#plt.show()
	plt.close()
	return
# ----------------------------------------------------------- #
# --- Scatter plot Ratios vs tau ---------------------------- #
# ----------------------------------------------------------- #
    def plot_ratio_vs_tau(self,LOESS = True, filename=None, fmt = 'pdf'):
	"""
	function to plot ratio vs tau plane together with LOESS smooth average curve

	Parameters:
	-----------
	LOESS:		bool, if True LOESS regression is plotted (R needs to be installed!)
	filename:	string, filename for output plot
	format:		string, format of output plot (pdf, eps, png)

	Returns
	-------
	Nothing
	"""

	import matplotlib
	import matplotlib.pyplot as plt

	fig = plt.figure(num = 1, figsize=(12,9))

# --- LOESS regression ---------------------------------------------------------------------#
	if LOESS:
	    fi = open('RvsT.dat','w')
	    for i,R in enumerate(self.ratio):
		fi.write('{0}\t{1}\n'.format(self.tau_ratio[i], R))
	    fi.close()

	    try:
		if os.environ["HOST"] == "uh2ulastro15":
		    subprocess.call(['R','-q','-f','script.r'])
		elif os.environ["HOST"] == "crf-wgs01" or os.environ["HOST"] == "astro-wgs01":
		    subprocess.call(['/nfs/astrop/d1/software/R/bin/R','-q','-f','script.r'])
		else:
		    logging.warning("R is not installed! Exit.")
	    except KeyError:
		try:
		    subprocess.call(['R','-q','-f','script.r'])
		except:
		    logging.warning("R is not installed! Exit.")
	    fi = open('loess.info')
	    for l in fi.readlines():
		if l.find('Equivalent') >= 0:
		    eff_num_params = float(l.split(':')[1])
		if l.find('Residual') >= 0:
		    res_err = float(l.split(':')[1])
		if l.find('Trace') >= 0:
		    trace_smooth = float(l.split(':')[1])
	    loess = np.loadtxt('loess.dat').transpose()

# --- Ratio vs Tau Plot ----------------------------------------------------------------#

	gs = matplotlib.gridspec.GridSpec(3,13)
	ax = plt.subplot(gs[:,:-1])
	cp = plt.cm.RdBu

	plt.axvline(self.TauThinLim, linestyle ='--', color = 'black', lw = 1.5)
	plt.axvline(self.TauThickLim, linestyle ='-', color = 'black', lw = 1.5)
	plt.axhline(0., linestyle ='-', color = '0.5', lw = 2.)


	for j,E in enumerate(self.energy_ratio):
	    plt.plot(self.tau_ratio[j],self.ratio[j],
		linestyle = 'None',
		marker = 'o',
		ms = (self.redshift_ratio[j]/0.536 * 5.)**1.5 + 4.5,
		color = cp( ( np.log10(E/0.1)/np.log10(10./0.1)) ),
		mec = '0.'
		)
	if LOESS:
	    plt.plot(np.sort(loess[0]),(loess[1])[np.argsort(self.tau_ratio)],lw = 3, color = 'black',label = 'Loess Regression')

    
# --- Set EBL String:
	ebl = ['Kneiske & Dole (2010)', 'Franceschini $et\,al.$ (2008)', 'Dominguez $et\,al.$ (2011)',\
	    'Inoue $et\,al.$ (2012)']
	if self.eblmodel== 'kneiske':
	    string = ebl[0]
	elif self.eblmodel == 'franceschini':
	    string = ebl[1]
	elif self.eblmodel == 'dominguez':
	    string = ebl[2]
	elif self.eblmodel == 'inoue':
	    string = ebl[3]
	plt.annotate('EBL: {0}, $\\alpha = {1}$\n$p_\\mathrm{{KS}} = {2:.2f}$'.format(string,self.TauScale,-ndtri(self.pKS)),
	    xy = (0.95,0.95), xycoords = 'axes fraction', ha = 'right', va = 'top', size = 'medium')
	plt.xlabel(r'Optical depth $\tau$', size = 'large')
	plt.ylabel('Ratio $R$', size = 'large')

	ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
	ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
	ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.2))
	ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.2))
	plot_extra_legend2(col = '0.5', lcol = 'black')

	plt.axis([0.,ceil(np.max(self.tau_ratio)),-1.1,1.1])
	ax1 = fig.add_axes([0.86, 0.1, 0.03, 0.8])

	# Set the colormap and norm to correspond to the data for which
	# the colorbar will be used.
	#norm = mpl.colors.Normalize(vmin=0, vmax=23.1)
	norm = mpl.colors.LogNorm(vmin=0.1, vmax=10.0)

	# ColorbarBase derives from ScalarMappable and puts a colorbar
	# in a specified axes, so it has everything needed for a
	# standalone colorbar.  There are many more kwargs, but the
	# following gives a basic continuous colorbar with ticks
	# and labels.
	cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cp,
	    norm=norm,
	    orientation='vertical',
	    ticks=np.hstack((np.linspace(0.1,1.,10),np.linspace(1.,10.,10)))#[10,1.,0.1]
	    )
	cb1.ax.set_yticklabels(['0.1','','','','','','','','','', '1','','','','','','','','','10'])
	cb1.set_label('Energy (TeV)',size='large')

	if filename == None:
	    plt.savefig('../plots/scatter_ratio_vs_tau_{0}_{1}.{2}'.format(self.eblmodel,str(self.TauScale).replace(".","p"),fmt), format = fmt)
	else:
	    plt.savefig(filename, format = fmt)

	#plt.show()
	plt.close()
	return

# ------------------------------------------------------------------------------------------#
# --- X-check plot: E (instead of tau) vs Residuals / Ratios -------------------------------#
# ------------------------------------------------------------------------------------------#
    def plot_residual_vs_E(self,LOESS = True, filename=None, ptype = 'residual', fmt = 'pdf'):
	"""
	function to plot residuals vs E together with LOESS smooth average curve

	Parameters:
	-----------
	LOESS:		bool, if True LOESS regression is plotted (R needs to be installed!)
	filename:	string, filename for output plot
	ptype:		string, either residual or ratio
	fmt:		string, format of output plot (pdf, eps, png)

	Returns
	-------
	Nothing
	"""

	import matplotlib
	import matplotlib.pyplot as plt

	fig = plt.figure()

	cp = plt.cm.copper

	if ptype == "residual":
	    energy = self.energy
	    rs = self.residual
	    zs = self.redshift
	    ylabel = "Residuals $\chi$"
	elif ptype == "ratio":
	    energy = self.energy_ratio
	    rs = self.ratio
	    zs = self.redshift_ratio
	    ylabel = "Ratios $R$"
	else:
	    logging.warning("unknwon ptype. Either residual or ratio allowed. returning -1")
	    return -1

# --- LOESS regression ---------------------------------------------------------------------#
	if LOESS:
	    fi = open('RvsE.dat','w')
	    for i,R in enumerate(rs):
		fi.write('{0}\t{1}\n'.format(np.log10(energy[i]), R))
	    fi.close()

	    try:
		if os.environ["HOST"] == "uh2ulastro15":
		    subprocess.call(['R','-q','-f','script_E.r'])
		elif os.environ["HOST"] == "crf-wgs01" or os.environ["HOST"] == "astro-wgs01":
		    subprocess.call(['/nfs/astrop/d1/software/R/bin/R','-q','-f','script_E.r'])
		else:
		    logging.warning("R is not installed! Exit.")
	    except KeyError:
		try:
		    subprocess.call(['R','-q','-f','script_E.r'])
		except:
		    logging.warning("R is not installed! Exit.")
	    fi = open('loess_E.info')
	    for l in fi.readlines():
		if l.find('Equivalent') >= 0:
		    eff_num_params = float(l.split(':')[1])
		if l.find('Residual') >= 0:
		    res_err = float(l.split(':')[1])
		if l.find('Trace') >= 0:
		    trace_smooth = float(l.split(':')[1])
	    loess = np.loadtxt('loess_E.dat').transpose()

# --- Residuals vs Tau Plot ----------------------------------------------------------------#

	ax = plt.subplot(111)
	ax.set_xscale('log')
	cp = plt.cm.RdBu

	plt.axhline(0., linestyle ='-', color = '0.5', lw = 2.)


	for j,E in enumerate(energy):
	    plt.plot(E,rs[j],
		linestyle = 'None',
		marker = 'o',
		ms = (zs[j]/0.536 * 5.)**1.5 + 4.5,
		color = '0.8',
		mec = '0.'
	    )
	if LOESS:
	    plt.plot(10.**np.sort(loess[0]),loess[1][np.argsort(energy)],lw = 3, color = 'red',label = 'Loess Regression')

    
# --- Set EBL String:
	ebl = ['Kneiske & Dole (2010)', 'Franceschini $et\,al.$ (2008)', 'Dominguez $et\,al.$ (2011)',\
	    'Inoue $et\,al.$ (2012)']
	if self.eblmodel== 'kneiske':
	    string = ebl[0]
	elif self.eblmodel == 'franceschini':
	    string = ebl[1]
	elif self.eblmodel == 'dominguez':
	    string = ebl[2]
	elif self.eblmodel == 'inoue':
	    string = ebl[3]
	plt.annotate('EBL: {0}, $\\alpha = {1}$'.format(string,self.TauScale),
	    xy = (0.95,0.95), xycoords = 'axes fraction', ha = 'right', va = 'top', size = 'x-large')
	plt.xlabel(r'Energy (TeV)', size = 'xx-large')
	plt.ylabel(ylabel, size = 'xx-large')

	for label in ax.xaxis.get_ticklabels():
	    label.set_fontsize('xx-large')
	for label in ax.yaxis.get_ticklabels():
	    label.set_fontsize('xx-large')


	ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
	ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.2))
	plot_extra_legend2()

	if ptype == 'residual':
	    plt.axis([np.min(self.energy)*0.9,np.max(self.energy)*1.1,-3.,4.])
	else:
	    plt.axis([np.min(self.energy)*0.9,np.max(self.energy)*1.1,-1.1,1.1])

	if filename == None:
	    plt.savefig('../plots/scatter_{3}_vs_E_{0}_{1}.{2}'.format(self.eblmodel,str(self.TauScale).replace(".","p"),fmt,ptype), format = fmt)
	else:
	    plt.savefig(filename, fmt = fmt)

	#plt.show()
	plt.close()
	return
