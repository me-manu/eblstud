"""
Module Containing tools for fitting with minuit migrad routine.
Based on python module iminuit

History of changes:
Version 0.1
- Created 9th Dec 2013
	* MinuitFitPL
	* MinuitFitLP
	* MinuitFitBPL
"""

# - Imports ------------------------- #
import numpy as np
from eblstud.tools.lsq_fit import *
import scipy.special
import iminuit as minuit
import warnings,logging
from math import ceil,floor
# ----------------------------------- #

logging.basicConfig(level=logging.WARNING)

# - Fitting functions ----------------------------------------------------------------------- #
pl      = lambda p,x : p['Prefactor'] * (x / p['Scale']) ** p['Index']
lp      = lambda p,x : p['norm'] * (x / p['Eb']) ** (p['alpha'] + p['beta'] * np.log(x / p['Eb']))
pl2     = lambda p,x : p['Integral'] * x ** p['Index'] * (p['Index'] + 1.) / (p['UpperLimit'] ** (p['Index'] + 1.) - p['LowerLimit'] ** (p['Index'] + 1.))
bpl_in	= lambda x,xb, f: 1. + np.power(x*np.abs(xb),f)
bpl	= lambda p, x: p['Prefactor']*(np.power(x / p['Scale'],p['Index1']))*np.power(bpl_in(x / p['Scale'],p['BreakValue'],p['Smooth']),(p['Index2'] - p['Index1'])/p['Smooth'])

# - Chi functions --------------------------------------------------------------------------- #
errfunc = lambda func, p, x, y, s: (func(p, x)-y) / s 

# - P-value --------------------------------------------------------------------------------- #
pvalue = lambda dof, chisq: 1. - gammainc(.5 * dof, .5 * chisq)

# - Butterfly functions --------------------------------------------------------------------- #
butterfly_pl = lambda p,x,s,cov : np.sqrt ((s['Prefactor']/p['Prefactor']) ** 2. + np.log(x/p['Scale'])**2. * s['Index'] ** 2. \
					+ 2. * cov['Index','Prefactor'] * np.log(x/p['Scale']) / p['Prefactor']             # Err^2 / Flux^2 for power law
					)
butterfly_lp = lambda p,x,s,cov : np.sqrt( (s['norm']/p['norm']) ** 2. \
					+ 2. * cov['norm','alpha'] * np.log(x/p['Eb']) / p['norm'] \
					+ np.log(x / p['Eb'] ) **2. * ( \
					s['alpha'] ** 2. + s['beta'] ** 2. * np.log(x/p['Eb']) ** 2. \
					+ 2. * (cov['norm','beta'] / p['norm'] + cov['alpha','beta'] * np.log(x/p['Eb']) ) \
					)
					)
def butterfly_bpl(p,x,s,cov):
    """
    Compute butterfly from matrix product between Jacobian and covariance matrix
    """
    inner		= bpl_in(x,p['BreakValue'],p['Smooth'])
    J = {}
    J['Prefactor']	= 1. / p['Prefactor'] * np.ones(x.shape[0])
    J['Index1']		= (np.log(x / p['Scale']) - np.log(inner) / p['Smooth'])
    J['Index2']		= np.log(inner) / p['Smooth']  
    J['BreakValue']	= (p['Index2'] - p['Index1']) / inner * x * (x * p['BreakValue']) ** (p['Smooth'] - 1.)
    J['Index2']	= np.zeros(x.shape[0])
    covar	= np.zeros((4,4))
    result = 0.
    for i,ki in enumerate(J.keys()):
	for j,kj in enumerate(J.keys()):
	    result += cov[ki,kj] * J[ki] * J[kj]
    return np.sqrt(result)



# - Power Law Fit --------------------------------------------------------------------------- #
def MinuitFitPL(x,y,s,full_output=False, **kwargs):
    """
    Function to fit Powerlaw to data using minuit.migrad

    y(x) = p['Prefactor'] * ( x / p['Scale'] ) ** p['Index']

    Parameters
    ----------
    x:	n-dim array containing the measured x values
    y:	n-dim array containing the measured y values, i.e. y = y(x)
    s:  n-dim array with (symmetric) measurment uncertainties on y

    kwargs
    ------
    full_output:	bool, if True, errors will be estimated additionally with minos, covariance matrix will also be returned
    print_level:	0,1, level of verbosity, defualt = 0 means nothing is printed
    int_steps:		float, initial step width, multiply with initial values of errors, default = 0.1
    strategy:		0 = fast, 1 = default (default), 2 = thorough
    tol:		float, required tolerance of fit = 0.001*tol*UP, default = 1.
    up			float, errordef, 1 (default) for chi^2, 0.5 for log-likelihood
    ncall		int, number of maximum calls, default = 1000
    pedantic		bool, if true (default), give all warnings
    limits		dictionary containing 2-tuple for all fit parameters
    pinit		dictionary with initial fit for all fir parameters
    fix			dictionary with booleans if parameter is frozen for all fit parameters

    Returns
    -------
    tuple containing
    	0. list of Fit Stats: ChiSq, Dof, P-value
    	1. dictionary with final fit parameters
    	2. dictionary with 1 Sigma errors of final fit parameters
    if full_output = True:
	3. dictionary with +/- 1 Sigma Minos errors
	4. dictionary with covariance matrix

    Notes
    -----
    iminuit documentation: http://iminuit.github.io/iminuit/index.html
    """
# --- Set the defaults
    kwargs.setdefault('print_level',0)		# no output
    kwargs.setdefault('int_steps',0.1)		# Initial step width, multiply with initial values in m.errors
    kwargs.setdefault('strategy',1)		# 0 = fast, 1 = default, 2 = thorough
    kwargs.setdefault('tol',1.)			# Tolerance of fit = 0.001*tol*UP
    kwargs.setdefault('up',1.)			# 1 for chi^2, 0.5 for log-likelihood
    kwargs.setdefault('ncall',1000.)		# number of maximum calls
    kwargs.setdefault('pedantic',True)		# Give all warnings
    kwargs.setdefault('limits',{})
    kwargs.setdefault('pinit',{})
    kwargs.setdefault('fix',{'Index': False , 'Prefactor': False, 'Scale' : True})	
# --------------------
	
    npar = 2

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    exp = floor(np.log10(y[0]))
    y /= 10.**exp
    s /= 10.**exp

    def FillChiSq(Prefactor,Index,Scale):
	params = {'Prefactor': Prefactor, 'Index': Index, 'Scale': Scale}
	return np.sum(errfunc(pl,params,x,y,s)**2.)

    # Set initial Fit parameters, initial step width and limits of parameters
    if not len(kwargs['pinit']):
	kwargs['pinit']['Scale']	= x[np.argmax(y/s)]
	kwargs['pinit']['Prefactor']	= prior_norm(x / kwargs['pinit']['Scale'],y)
	kwargs['pinit']['Index']	= prior_pl_ind(x / kwargs['pinit']['Scale'],y)
    else:
	kwargs['pinit']['Prefactor'] /= 10.**exp
    if not len(kwargs['limits']):
	kwargs['limits']['Prefactor'] = (kwargs['pinit']['Prefactor'] / 1e2, kwargs['pinit']['Prefactor'] * 1e2)
	kwargs['limits']['Index'] = (-10.,10.)
	kwargs['limits']['Scale'] = (kwargs['pinit']['Scale'] / 1e2, kwargs['pinit']['Scale'] * 1e2)
    

    m = minuit.Minuit(FillChiSq, print_level = kwargs['print_level'],
			# initial values
			Prefactor	= kwargs['pinit']["Prefactor"],
			Index = kwargs['pinit']["Index"],
			Scale = kwargs['pinit']["Scale"],
			# errors
			error_Prefactor	= kwargs['pinit']['Prefactor'] * kwargs['int_steps'],
			error_Index	= kwargs['pinit']['Index'] * kwargs['int_steps'],
			error_Scale	= 0.,
			# limits
			limit_Prefactor = kwargs['limits']['Prefactor'],
			limit_Index	= kwargs['limits']['Index'],
			limit_Scale	= kwargs['limits']['Scale'],
			# freeze parametrs 
			fix_Prefactor	= kwargs['fix']['Prefactor'],
			fix_Index	= kwargs['fix']['Index'],
			fix_Scale	= kwargs['fix']['Scale'],
			# setup
			pedantic	= kwargs['pedantic'],
			errordef	= kwargs['up'],
			)

    # Set initial fit control variables
    m.tol	= kwargs['tol']
    m.strategy	= kwargs['strategy']

    m.migrad(ncall = kwargs['ncall'])
    # second fit
    m = minuit.Minuit(FillChiSq, print_level = kwargs['print_level'],errordef = kwargs['up'], **m.fitarg)
    m.migrad(ncall = kwargs['ncall'])
    logging.debug("PL: Migrad minimization finished")

    m.hesse()
    logging.debug("PL: Hesse matrix calculation finished")

    if full_output:
	logging.debug("PL: Running Minos for error estimation")
	for k in kwargs['pinit'].keys():
	    if kwargs['fix'][k]:
		continue
	    m.minos(k,1.)
	logging.debug("PL: Minos finished")

    fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)

    m.values['Prefactor'] *= 10.**exp
    m.errors['Prefactor'] *= 10.**exp

    try:
	for k in kwargs['pinit'].keys():
	    if kwargs['fix'][k]:
		continue
	    m.covariance[k,'Prefactor'] *= 10.**exp 
	    m.covariance['Prefactor',k] *= 10.**exp 
    except TypeError:
	logging.warning("Hesse matrix not available. Did iminuit.hesse fail?")

    if full_output:
	return fit_stat,m.values, m.errors,m.merrors, m.covariance
    else:	
	return fit_stat,m.values, m.errors


# - Logarithmic Parabola Fit----------------------------------------------------------------- #
def MinuitFitLP(x,y,s,full_output=False, **kwargs):
    """
    Function to fit Logarithmic Parabola to data using minuit.migrad

    y(x) = p['norm'] * ( x / p['Eb'] ) ** (p['alpha'] + p['beta'] * ln( x / p['Eb']))

    Parameters
    ----------
    x:	n-dim array containing the measured x values
    y:	n-dim array containing the measured y values, i.e. y = y(x)
    s:  n-dim array with (symmetric) measurment uncertainties on y

    kwargs
    ------
    full_output:	bool, if True, errors will be estimated additionally with minos, covariance matrix will also be returned
    print_level:	0,1, level of verbosity, defualt = 0 means nothing is printed
    int_steps:		float, initial step width, multiply with initial values of errors, default = 0.1
    strategy:		0 = fast, 1 = default (default), 2 = thorough
    tol:		float, required tolerance of fit = 0.001*tol*UP, default = 1.
    up			float, errordef, 1 (default) for chi^2, 0.5 for log-likelihood
    ncall		int, number of maximum calls, default = 1000
    pedantic		bool, if true (default), give all warnings
    limits		dictionary containing 2-tuple for all fit parameters
    pinit		dictionary with initial fit for all fir parameters
    fix			dictionary with booleans if parameter is frozen for all fit parameters
    func		function pointer to analytical function. Needs to be of the form func(params,E), where 
    			params = {'Eb','alpha','beta',norm'} (default = lp)

    Returns
    -------
    tuple containing
    	0. list of Fit Stats: ChiSq, Dof, P-value
    	1. dictionary with final fit parameters
    	2. dictionary with 1 Sigma errors of final fit parameters
    if full_output = True:
	3. dictionary with +/- 1 Sigma Minos errors
	4. dictionary with covariance matrix

    Notes
    -----
    iminuit documentation: http://iminuit.github.io/iminuit/index.html
    """
# --- Set the defaults
    kwargs.setdefault('print_level',0)		# no output
    kwargs.setdefault('func',lp)		# fitting function
    kwargs.setdefault('int_steps',0.1)		# Initial step width, multiply with initial values in m.errors
    kwargs.setdefault('strategy',1)		# 0 = fast, 1 = default, 2 = thorough
    kwargs.setdefault('tol',1.)			# Tolerance of fit = 0.001*tol*UP
    kwargs.setdefault('up',1.)			# 1 for chi^2, 0.5 for log-likelihood
    kwargs.setdefault('ncall',1000.)		# number of maximum calls
    kwargs.setdefault('pedantic',True)		# Give all warnings
    kwargs.setdefault('limits',{})
    kwargs.setdefault('pinit',{})
    kwargs.setdefault('fix',{'alpha': False , 'beta': False, 'norm': False, 'Eb' : True})	
# --------------------
    npar = 3


    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    exp = floor(np.log10(y[0]))
    y /= 10.**exp
    s /= 10.**exp

    def FillChiSq(norm,alpha,beta,Eb):
	params = {'norm': norm, 'alpha': alpha, 'Eb': Eb, 'beta': beta}
	return np.sum(errfunc(kwargs['func'],params,x,y,s)**2.)

    # Set initial Fit parameters, initial step width and limits of parameters
    if not len(kwargs['pinit']):
	kwargs['pinit']['Eb']	= x[np.argmax(y/s)]
	kwargs['pinit']['norm']	= prior_norm(x / kwargs['pinit']['Eb'],y)
	kwargs['pinit']['alpha'],kwargs['pinit']['beta'] = prior_logpar(x / kwargs['pinit']['Eb'],y)
    else:
	kwargs['pinit']['norm'] /= 10.**exp
    if not len(kwargs['limits']):
	kwargs['limits']['norm'] = (kwargs['pinit']['norm'] / 1e2, kwargs['pinit']['norm'] * 1e2)
	kwargs['limits']['alpha'] = (-10.,2.)
	#kwargs['limits']['beta'] = (-5.,5.)
	kwargs['limits']['beta'] = (-5.,0.5)
	kwargs['limits']['Eb'] = (kwargs['pinit']['Eb'] / 1e2, kwargs['pinit']['Eb'] * 1e2)
    
    m = minuit.Minuit(FillChiSq,
			# initial values
			norm	= kwargs['pinit']["norm"],
			alpha	= kwargs['pinit']["alpha"],
			beta	= kwargs['pinit']["beta"],
			Eb	= kwargs['pinit']["Eb"],
			# errors
			error_norm	= kwargs['pinit']['norm'] * kwargs['int_steps'],
			error_alpha	= kwargs['pinit']['alpha'] * kwargs['int_steps'],
			error_beta	= kwargs['pinit']['beta'] * kwargs['int_steps'],
			error_Eb	= 0.,
			# limits
			limit_norm	= kwargs['limits']['norm'],
			limit_alpha	= kwargs['limits']['alpha'],
			limit_beta	= kwargs['limits']['beta'],
			limit_Eb	= kwargs['limits']['Eb'],
			# freeze parametrs 
			fix_norm	= kwargs['fix']['norm'],
			fix_alpha	= kwargs['fix']['alpha'],
			fix_beta	= kwargs['fix']['beta'],
			fix_Eb		= kwargs['fix']['Eb'],
			# setup
			print_level = kwargs['print_level'],
			pedantic	= kwargs['pedantic'],
			errordef	= kwargs['up'],
			)

    # Set initial fit control variables
    m.tol	= kwargs['tol']
    m.strategy	= kwargs['strategy']

    m.migrad(ncall = kwargs['ncall'])
    # second fit
    m = minuit.Minuit(FillChiSq, print_level = kwargs['print_level'],errordef = kwargs['up'], **m.fitarg)
    m.migrad(ncall = kwargs['ncall'])
    logging.debug("LP: Migrad minimization finished")

    m.hesse()
    logging.debug("LP: Hesse matrix calculation finished")

    if full_output:
	logging.debug("LP: Running Minos for error estimation")
	for k in kwargs['pinit'].keys():
	    if kwargs['fix'][k]:
		continue
	    m.minos(k,1.)
	logging.debug("LP: Minos finished")

    fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)

    m.values['norm'] *= 10.**exp
    m.errors['norm'] *= 10.**exp

    try:
	for k in kwargs['pinit'].keys():
	    if kwargs['fix'][k]:
		continue
	    m.covariance['norm',k] *= 10.**exp 
	    m.covariance[k,'norm'] *= 10.**exp 
    except TypeError:
	logging.warning("Hesse matrix not available. Did iminuit.hesse fail?")

    if full_output:
	return fit_stat,m.values, m.errors,m.merrors, m.covariance
    else:	
	return fit_stat,m.values, m.errors


# - Broken Power law Fit -------------------------------------------------------------------- #
def MinuitFitBPL(x,y,s,full_output=False, **kwargs):
    """
    Function to fit Broken Power Law with smooth transition to data using minuit.migrad

    y(x) = p['Prefactor'] * ( x / p['Scale'] ) ** p['Index1'] * ( 1. + x * p['BreakValue'] ** p['Smooth'] ) ** ((p['Index2'] - p['Index1']) / p['Smooth'])

    Parameters
    ----------
    x:	n-dim array containing the measured x values
    y:	n-dim array containing the measured y values, i.e. y = y(x)
    s:  n-dim array with (symmetric) measurment uncertainties on y

    kwargs
    ------
    full_output:	bool, if True, errors will be estimated additionally with minos, covariance matrix will also be returned
    print_level:	0,1, level of verbosity, defualt = 0 means nothing is printed
    int_steps:		float, initial step width, multiply with initial values of errors, default = 0.1
    strategy:		0 = fast, 1 = default (default), 2 = thorough
    tol:		float, required tolerance of fit = 0.001*tol*UP, default = 1.
    up			float, errordef, 1 (default) for chi^2, 0.5 for log-likelihood
    ncall		int, number of maximum calls, default = 1000
    pedantic		bool, if true (default), give all warnings
    limits		dictionary containing 2-tuple for all fit parameters
    pinit		dictionary with initial fit for all fir parameters
    fix			dictionary with booleans if parameter is frozen for all fit parameters

    Returns
    -------
    tuple containing
    	0. list of Fit Stats: ChiSq, Dof, P-value
    	1. dictionary with final fit parameters
    	2. dictionary with 1 Sigma errors of final fit parameters
    if full_output = True:
	3. dictionary with +/- 1 Sigma Minos errors
	4. dictionary with covariance matrix

    Notes
    -----
    iminuit documentation: http://iminuit.github.io/iminuit/index.html
    """
# --- Set the defaults
    kwargs.setdefault('print_level',0)		# no output
    kwargs.setdefault('int_steps',0.1)		# Initial step width, multiply with initial values in m.errors
    kwargs.setdefault('strategy',1)		# 0 = fast, 1 = default, 2 = thorough
    kwargs.setdefault('tol',1.)			# Tolerance of fit = 0.001*tol*UP
    kwargs.setdefault('up',1.)			# 1 for chi^2, 0.5 for log-likelihood
    kwargs.setdefault('ncall',1000.)		# number of maximum calls
    kwargs.setdefault('pedantic',True)		# Give all warnings
    kwargs.setdefault('limits',{})
    kwargs.setdefault('pinit',{})
    kwargs.setdefault('fix',{'Index1': False ,'Index2': False , 'Smooth': True, 'BreakValue': False, 'Prefactor': False, 'Scale' : True})	
# --------------------
	
    npar = 4

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    exp = floor(np.log10(y[0]))
    y /= 10.**exp
    s /= 10.**exp

    def FillChiSq(Prefactor,Index1,Index2,BreakValue,Smooth,Scale):
	params = {'Prefactor': Prefactor, 'Index1': Index1, 'Index2': Index2, 'BreakValue': BreakValue, 'Smooth': Smooth, 'Scale': Scale}
	return np.sum(errfunc(bpl,params,x,y,s)**2.)

    # Set initial Fit parameters, initial step width and limits of parameters
    if not len(kwargs['pinit']):
	kwargs['pinit']['Scale']	= x[np.argmax(y/s)]
	kwargs['pinit']['Prefactor']	= prior_norm(x / kwargs['pinit']['Scale'],y)
	kwargs['pinit']['Index1'],kwargs['pinit']['Index2'],kwargs['pinit']['BreakValue']	= prior_bpl(x / kwargs['pinit']['Scale'],y)
	kwargs['pinit']['Smooth']	= 4.
    else:
	kwargs['pinit']['Prefactor'] /= 10.**exp
    if not len(kwargs['limits']):
	kwargs['limits']['Prefactor'] = (kwargs['pinit']['Prefactor'] / 1e2, kwargs['pinit']['Prefactor'] * 1e2)
	kwargs['limits']['Index1'] = (-10.,2.)
	kwargs['limits']['Index2'] = (-10.,2.)
	kwargs['limits']['BreakValue'] = (kwargs['pinit']['Scale'] / x[-1] , kwargs['pinit']['Scale'] / x[0]  )
	kwargs['limits']['Scale'] = (kwargs['pinit']['Scale'] / 1e2, kwargs['pinit']['Scale'] * 1e2)
	kwargs['limits']['Smooth'] = (0.5,10.)
    

    m = minuit.Minuit(FillChiSq, print_level = kwargs['print_level'],
			# initial values
			Prefactor	= kwargs['pinit']["Prefactor"],
			Index1		= kwargs['pinit']["Index1"],
			Index2		= kwargs['pinit']["Index2"],
			BreakValue	= kwargs['pinit']["BreakValue"],
			Scale		= kwargs['pinit']["Scale"],
			Smooth		= kwargs['pinit']["Smooth"],
			# errors
			error_Prefactor	= kwargs['pinit']['Prefactor'] * kwargs['int_steps'],
			error_Index1	= kwargs['pinit']['Index1'] * kwargs['int_steps'],
			error_Index2	= kwargs['pinit']['Index2'] * kwargs['int_steps'],
			error_BreakValue= kwargs['pinit']['BreakValue'] * kwargs['int_steps'],
			error_Scale	= 0.,
			error_Smooth	= 0.,
			# limits
			limit_Prefactor = kwargs['limits']['Prefactor'],
			limit_Index1	= kwargs['limits']['Index1'],
			limit_Index2	= kwargs['limits']['Index2'],
			limit_BreakValue= kwargs['limits']['BreakValue'],
			limit_Scale	= kwargs['limits']['Scale'],
			limit_Smooth	= kwargs['limits']['Smooth'],
			# freeze parametrs 
			fix_Prefactor	= kwargs['fix']['Prefactor'],
			fix_Index1	= kwargs['fix']['Index1'],
			fix_Index2	= kwargs['fix']['Index2'],
			fix_BreakValue	= kwargs['fix']['BreakValue'],
			fix_Scale	= kwargs['fix']['Scale'],
			fix_Smooth	= kwargs['fix']['Smooth'],
			# setup
			pedantic	= kwargs['pedantic'],
			errordef	= kwargs['up']
			)

    # Set initial fit control variables
    m.tol	= kwargs['tol']
    m.strategy	= kwargs['strategy']

    m.migrad(ncall = kwargs['ncall'])
    # second fit
    m = minuit.Minuit(FillChiSq, print_level = kwargs['print_level'],errordef = kwargs['up'], **m.fitarg)
    m.migrad(ncall = kwargs['ncall'])
    logging.debug("BPL: Migrad minimization finished")

    m.hesse()
    logging.debug("BPL: Hesse matrix calculation finished")

    if full_output:
	logging.debug("BPL: Running Minos for error estimation")
	for k in kwargs['pinit'].keys():
	    if kwargs['fix'][k]:
		continue
	    m.minos(k,1.)
	logging.debug("BPL: Minos finished")

    fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)

    m.values['Prefactor'] *= 10.**exp
    m.errors['Prefactor'] *= 10.**exp

    try:
	for k in kwargs['pinit'].keys():
	    if kwargs['fix'][k]:
		continue
	    m.covariance[k,'Prefactor'] *= 10.**exp 
	    m.covariance['Prefactor',k] *= 10.**exp 
    except TypeError:
	logging.warning("Hesse matrix not available. Did iminuit.hesse fail?")


    if full_output:
	return fit_stat,m.values, m.errors,m.merrors, m.covariance
    else:	
	return fit_stat,m.values, m.errors
