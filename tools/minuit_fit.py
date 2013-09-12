"""
Module Containing tools for fitting with minuit migrad routine.

History of changes:
Version 1.0
- Created 9th May 2011
- 1st June: Added models, functions now are
	* MinuitFitPL
	* MinuitFitLP
	* MinuitFitEPL
	* MinuitFitBPL
	* MinuitFitEBPL
	* MinuitFitDBPL
	* MinuitFitEDBPL
"""

import numpy as np
from eblstud.stats import misc
from eblstud.tools.lsq_fit import *
import scipy.special
import minuit
import warnings

class FitMinuit:
    def __init__(self,minos = False):
# - Set Minuit Fit parameters --------------------------------------------------------------- #
	self.FitParams = {
	    'int_steps' : 0.1,		# Initial step width, multiply with initial values in m.errors
	    'strategy'  : 0,		# 0 = fast, 1 = default, 2 = thorough
	    'printMode' : 0,		# Shut Minuit up
	    'maxcalls'  : 1000,		# Maximum Number of Function calls, default is None
	    'tol'       : 0.1,		# Tolerance of fit = 0.001*tol*fit
	    'up'        : 1.,		# 1 for chi^2, 0.5 for log-likelihood
	    'SpecIndLoLim': -15.,	# Lower limit for spectral index
	    'SpecIndUpLim': 35.,	# Upper limit for spectral index
	    'SupExpLoLim' : 2.,		# Lower limit for super exponential parameter
	    'SupExpUpLim' : 15.		# Upper limit for super exponential parameter
	}
	self.minos = minos
	return
# - Initial Fit Parameters ------------------------------------------------------------------ #
    def SetFitParams(self,Minuit):
	Minuit.maxcalls = self.FitParams['maxcalls']
	Minuit.strategy = self.FitParams['strategy']
	Minuit.tol	= self.FitParams['tol']
	Minuit.up	= self.FitParams['up']
	Minuit.printMode= self.FitParams['printMode']
	return

# - Power Law Fit --------------------------------------------------------------------------- #
    def MinuitFitPL(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit Powerlaw to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 2
	fitfunc = errfunc_pl

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1):
	    params = N0,G1
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'] = prior_pl_ind(x,y)
	else:
	    m.values['N0'] = pinit[0]
	    m.values['G1'] = pinit[1]
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    pfinal[i] = m.values[val]
	    fit_err[i] = m.errors[val]
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err

# - Gaussian Fit --------------------------------------------------------------------------- #
    def MinuitFitGaussian(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit logartithmic parabola to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = Mean
	pinit[2] = 1./Sqrt(Var)
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 3
	fitfunc = errfunc_gaussian

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,Mean,Var):
	    params = N0,Mean,Var
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = 1.
	    m.values['Mean'], m.values['Var'] = prior_gaussian(x)
	else:
	    m.values['N0'],m.values['Mean'],m.values['Var'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['Mean'] = limits[i]
		if i == 2:
		    m.limits['Var'] = limits[i]
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('Mean',1.)
		m.minos('Var',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['Mean']
		fit_err[i] = m.errors['Mean']
	    if i == 2:
		pfinal[i] = m.values['Var']
		fit_err[i] = m.errors['Var']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Logarithmic Parabola Fit----------------------------------------------------------------- #
    def MinuitFitLP(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit logartithmic parabola to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = Curvature
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 3
	fitfunc = errfunc_logpar

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1,Curv):
	    params = N0,G1,Curv
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'], m.values['Curv'] = prior_logpar(x,y)
	else:
	    m.values['N0'],m.values['G1'],m.values['Curv'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['Curv'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['Curv'] = (-2.,2.)
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('Curv',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['Curv']
		fit_err[i] = m.errors['Curv']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Logarithmic Parabola Fit with exp. pile-up / Cut-off ------------------------------------- #
    def MinuitFitELP(self,x,y,s,pinit=[],limits=()):
	"""Function to fit logartithmic parabola to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = Curvature
	pinit[3] = 1. / Cut-off or Pile Up Energy
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 4
	fitfunc = errfunc_elogpar

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1,Curv,Cut):
	    params = N0,G1,Curv,Cut
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'], m.values['Curv'] = prior_logpar(x,y)
	    m.values['Cut'] = prior_epl_cut(x,y)
	else:
	    m.values['N0'],m.values['G1'],m.values['Curv'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['Curv'] = limits[i]
		if i == 3:
		    m.limits['Cut'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['Curv'] = (-2.,2.)
	    #m.limits['Cut'] = (0.,2./(x[0]))
	    m.limits['Cut'] = (1./(2.*x[-1]),2./(x[0]))
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('Curv',1.)
		m.minos('Cut',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['Curv']
		fit_err[i] = m.errors['Curv']
	    if i == 3:
		pfinal[i] = m.values['Cut']
		fit_err[i] = m.errors['Cut']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Power Law with Exp Pile up / cut off--------------------------------------------------------- #
    def MinuitFitEPL(self,x,y,s,pinit=[],limits=()):
	"""Function to fit Powerlaw with super exponential pile-up / cut-off 
	to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = 1./ break energy 
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 3
	fitfunc = errfunc_pl

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	supexp = 5.
	def FillChiSq(N0,G1,Cut):
	    params = N0,G1,Cut
	    result = 0.
	    for i,xval in enumerate(x):
		result += ((N0*(np.power(xval,G1))*np.exp(np.power(xval*Cut,supexp)) - y[i])\
		    / s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'] = prior_epl_ind(x,y)
	    m.values['Cut'] = prior_epl_cut(x,y)
	else:
	    m.values['N0'],m.values['G1'],m.values['Cut'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['Cut'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    #m.limits['Cut'] = (-2./(x[0]),2./(x[0]))
	    #m.limits['Cut'] = (0.,2./(x[0]))
	    m.limits['Cut'] = (1./(2.*x[-1]),2./(x[0]))
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('Cut',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['Cut']
		fit_err[i] = m.errors['Cut']
	return fit_stat,pfinal,fit_err
# - Broken Power law Fit -------------------------------------------------------------------- #
    def MinuitFitBPL(self,x,y,s,pinit=[],limits=(),full_output = False):
	"""Function to fit broken Powerlaw to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = 2nd pl index
	pinit[3] = 1./ break energy 
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 4
	fitfunc = errfunc_bpl

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1,G2,Bre):
	    params = N0,G1,G2,Bre
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'], m.values['G2'],m.values['Bre'] = prior_bpl(x,y)
	else:
	    m.values['N0'],m.values['G1'],m.values['G2'],m.values['Bre'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['G2'] = limits[i]
		if i == 3:
		    m.limits['Bre'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G2'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['Bre'] = (1./(2.*x[-1]),2./(x[0]))
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('G2',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['G2']
		fit_err[i] = m.errors['G2']
	    if i == 3:
		pfinal[i] = m.values['Bre']
		fit_err[i] = m.errors['Bre']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:
	    return fit_stat,pfinal,fit_err
# - Broken Power law with exp. pile up / cut-off--------------------------------------------- #
    def MinuitFitEBPL(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit broken Powerlaw to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = 2nd pl index
	pinit[3] = 1./ break energy 
	pinit[4] = 1./ pile-up/cut-off energy 
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 5
	fitfunc = errfunc_ebpl

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1,G2,Bre,Cut):
	    params = N0,G1,G2,Bre,Cut
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'], m.values['G2'],m.values['Bre'] = prior_bpl(x,y)
	    m.values['Cut'] = prior_epl_cut(x,y)
	else:
	    m.values['N0'],m.values['G1'],m.values['G2'],m.values['Bre'],m.values['Cut'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['G2'] = limits[i]
		if i == 3:
		    m.limits['Bre'] = limits[i]
		if i == 4:
		    m.limits['Cut'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G2'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['Bre'] = (1./(2.*x[-1]),2./(x[0]))
	    m.limits['Cut'] = (1./(2.*x[-1]),2./(x[0]))
	    #m.limits['Cut'] = (0.,2./(x[0]))
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('G2',1.)
		m.minos('Cut',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['G2']
		fit_err[i] = m.errors['G2']
	    if i == 3:
		pfinal[i] = m.values['Bre']
		fit_err[i] = m.errors['Bre']
	    if i == 4:
		pfinal[i] = m.values['Cut']
		fit_err[i] = m.errors['Cut']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Broken Power law with exp. pile up / cut-off that is not fixed but fitted --------------- #
    def MinuitFitEBPL_FS(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit broken Powerlaw to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = 2nd pl index
	pinit[3] = 1./ break energy 
	pinit[4] = 1./ pile-up/cut-off energy 
	pinit[5] = sup exp pile up parameter >= 1
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 6
	fitfunc = errfunc_sebpl_fs

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1,G2,Bre,Cut,SupExp):
	    params = N0,G1,G2,Bre,Cut,SupExp
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'], m.values['G2'],m.values['Bre'] = prior_bpl(x,y)
	    m.values['Cut'] = prior_epl_cut(x,y)
	    m.values['SupExp'] = 2.
	else:
	    m.values['N0'],m.values['G1'],m.values['G2'],m.values['Bre'],\
		m.values['Cut'],m.values['SupExp'] = pinit
	    if m.values['SupExp'] < 1.:
		m.values['SupExp'] = 2.
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['G2'] = limits[i]
		if i == 3:
		    m.limits['Bre'] = limits[i]
		if i == 4:
		    m.limits['Cut'] = limits[i]
		if i == 4:
		    m.limits['SupExp'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G2'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['Bre'] = (1./(2.*x[-1]),2./(x[0]))
	    m.limits['Cut'] = (1./(2.*x[-1]),2./(x[0]))
	    m.limits['SupExp'] = (self.FitParams['SupExpLoLim'],self.FitParams['SupExpUpLim'])
	    #m.limits['Cut'] = (0.,2./(x[0]))
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	#m.fixed['SupExp'] = True

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('G2',1.)
		m.minos('Cut',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['G2']
		fit_err[i] = m.errors['G2']
	    if i == 3:
		pfinal[i] = m.values['Bre']
		fit_err[i] = m.errors['Bre']
	    if i == 4:
		pfinal[i] = m.values['Cut']
		fit_err[i] = m.errors['Cut']
	    if i == 5:
		pfinal[i] = m.values['SupExp']
		fit_err[i] = m.errors['SupExp']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Double Broken Power Law ------------------------------------------------------------------- #
    def MinuitFitDBPL(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit Powerlaw with super exponential pile-up / cut-off 
	to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = pl index 2
	pinit[3] = pl index 3
	pinit[4] = 1./ break energy
	pinit[5] = 1./ break energy 2 
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 6
	fitfunc = errfunc_dbpl

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1,G2,G3,Bre,Bre2):
	    params = N0,G1,G2,G3,Bre,Bre2
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'], m.values['G2'],m.values['G3'],m.values['Bre'],m.values['Bre2'] \
		= prior_dbpl(x,y)
	else:
	    m.values['N0'],m.values['G1'], m.values['G2'],\
	    m.values['G3'],m.values['Bre'],m.values['Bre2'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['G2'] = limits[i]
		if i == 3:
		    m.limits['G3'] = limits[i]
		if i == 4:
		    m.limits['Bre'] = limits[i]
		if i == 5:
		    m.limits['Bre2'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G2'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G3'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['Bre'] = (1./(2.*x[-1]),2./(x[0]))
	    m.limits['Bre2'] = (1./(2.*x[-1]),2./(x[0]))
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('G2',1.)
		m.minos('G3',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['G2']
		fit_err[i] = m.errors['G2']
	    if i == 3:
		pfinal[i] = m.values['G3']
		fit_err[i] = m.errors['G3']
	    if i == 4:
		pfinal[i] = m.values['Bre']
		fit_err[i] = m.errors['Bre']
	    if i == 5:
		pfinal[i] = m.values['Bre2']
		fit_err[i] = m.errors['Bre2']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Double Broken Power Law with super exponential cut-off / pile-up -------------------------- #
    def MinuitFitEDBPL(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit double broken Powerlaw with super exponential pile-up / cut-off 
	to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = pl index 2
	pinit[3] = pl index 3
	pinit[4] = 1./ break energy
	pinit[5] = 1./ break energy 2 
	pinit[6] = 1./ cut-off / pile-up 2 
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 7
	fitfunc = errfunc_edbpl

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1,G2,G3,Bre,Bre2,Cut):
	    params = N0,G1,G2,G3,Bre,Bre2,Cut
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'], m.values['G2'],m.values['G3'],m.values['Bre'],m.values['Bre2'] \
		= prior_dbpl(x,y)
	    m.values['Cut'] = prior_epl_cut(x,y)
	else:
	    m.values['N0'],m.values['G1'], m.values['G2'],\
	    m.values['G3'],m.values['Bre'],m.values['Bre2'],m.values['Cut'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['G2'] = limits[i]
		if i == 3:
		    m.limits['G3'] = limits[i]
		if i == 4:
		    m.limits['Bre'] = limits[i]
		if i == 5:
		    m.limits['Bre2'] = limits[i]
		if i == 6:
		    m.limits['Cut'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G2'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G3'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['Bre'] = (1./(2.*x[-1]),2./(x[0]))
	    m.limits['Bre2'] = (1./(2.*x[-1]),2./(x[0]))
	    #m.limits['Cut'] = (0.,2./(x[0]))
	    m.limits['Cut'] = (1./(2.*x[-1]),2./(x[0]))
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('G2',1.)
		m.minos('G3',1.)
		m.minos('Cut',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['G2']
		fit_err[i] = m.errors['G2']
	    if i == 3:
		pfinal[i] = m.values['G3']
		fit_err[i] = m.errors['G3']
	    if i == 4:
		pfinal[i] = m.values['Bre']
		fit_err[i] = m.errors['Bre']
	    if i == 5:
		pfinal[i] = m.values['Bre2']
		fit_err[i] = m.errors['Bre2']
	    if i == 6:
		pfinal[i] = m.values['Cut']
		fit_err[i] = m.errors['Cut']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Double Broken Power Law with super exponential cut-off / pile-up and free Sup Exp parameter #
    def MinuitFitEDBPL_FS(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit double broken Powerlaw with super exponential pile-up / cut-off 
	to data using minuit.migrad
    
	pinit[0] = norm
	pinit[1] = pl index
	pinit[2] = pl index 2
	pinit[3] = pl index 3
	pinit[4] = 1./ break energy
	pinit[5] = 1./ break energy 2 
	pinit[6] = 1./ cut-off / pile-up 2 
	pinit[7] = Super exponential parameter
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 8
	fitfunc = errfunc_edbpl_fs

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(N0,G1,G2,G3,Bre,Bre2,Cut,SupExp):
	    params = N0,G1,G2,G3,Bre,Bre2,Cut,SupExp
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['N0'] = prior_norm(x,y)
	    m.values['G1'], m.values['G2'],m.values['G3'],m.values['Bre'],m.values['Bre2'] \
		= prior_dbpl(x,y)
	    m.values['Cut'] = prior_epl_cut(x,y)
	    m.values['SupExp'] = 2.5
	else:
	    m.values['N0'],m.values['G1'], m.values['G2'],\
	    m.values['G3'],m.values['Bre'],m.values['Bre2'],\
	    m.values['Cut'],m.values['SupExp'] = pinit
	    if m.values['SupExp'] < 1.:
		m.values['SupExp'] = 2.5
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['N0'] = limits[i]
		if i == 1:
		    m.limits['G1'] = limits[i]
		if i == 2:
		    m.limits['G2'] = limits[i]
		if i == 3:
		    m.limits['G3'] = limits[i]
		if i == 4:
		    m.limits['Bre'] = limits[i]
		if i == 5:
		    m.limits['Bre2'] = limits[i]
		if i == 6:
		    m.limits['Cut'] = limits[i]
		if i == 7:
		    m.limits['SupExp'] = limits[i]
	else:
	    m.limits['G1'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G2'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['G3'] = (self.FitParams['SpecIndLoLim'],self.FitParams['SpecIndUpLim'])
	    m.limits['Bre'] = (1./(2.*x[-1]),2./(x[0]))
	    m.limits['Bre2'] = (1./(2.*x[-1]),2./(x[0]))
	    #m.limits['Cut'] = (0.,2./(x[0]))
	    m.limits['Cut'] = (1./(2.*x[-1]),2./(x[0]))
	    m.limits['SupExp'] = (self.FitParams['SupExpLoLim'],self.FitParams['SupExpUpLim'])
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	#m.fixed['SupExp'] = True
	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('G1',1.)
		m.minos('G2',1.)
		m.minos('G3',1.)
		m.minos('Cut',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['N0']
		fit_err[i] = m.errors['N0']
	    if i == 1:
		pfinal[i] = m.values['G1']
		fit_err[i] = m.errors['G1']
	    if i == 2:
		pfinal[i] = m.values['G2']
		fit_err[i] = m.errors['G2']
	    if i == 3:
		pfinal[i] = m.values['G3']
		fit_err[i] = m.errors['G3']
	    if i == 4:
		pfinal[i] = m.values['Bre']
		fit_err[i] = m.errors['Bre']
	    if i == 5:
		pfinal[i] = m.values['Bre2']
		fit_err[i] = m.errors['Bre2']
	    if i == 6:
		pfinal[i] = m.values['Cut']
		fit_err[i] = m.errors['Cut']
	    if i == 7:
		pfinal[i] = m.values['SupExp']
		fit_err[i] = m.errors['SupExp']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Fit Function provided by Fermi tools to optical depth ------------------------------------- #
    def MinuitFitTau_Fermi(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function fitted to optical depth
	Function given by Fermi tools:
	tau(E) = (E - Eb) * p0 + p1 * ln(E/Eb) + p2* ln(E/Eb)**2
	Energies in MeV
    
	pinit[0] = Eb
	pinit[1] = p0
	pinit[2] = p1
	pinit[2] = p2
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 4
	fitfunc = errfunc_tau_fermi

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(Eb,p0,p1,p2):
	    params = Eb,p0,p1,p2
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['Eb'] = 1e4	# Eb is 10 GeV
	    m.values['p0'] = 0.
	    m.values['p1'] = 1.
	    m.values['p2'] = 0.5
	else:
	    m.values['Eb'],m.values['p0'], m.values['p1'],\
	    m.values['p2'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['Eb'] = limits[i]
		if i == 1:
		    m.limits['p0'] = limits[i]
		if i == 2:
		    m.limits['p1'] = limits[i]
		if i == 3:
		    m.limits['p2'] = limits[i]
	else:
	    m.limits['Eb'] = (0.001,0.05)
	    m.limits['p0'] = (-10.,10.)
	    m.limits['p1'] = (-10.,10.)
	    m.limits['p2'] = (-10.,10.)
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	#m.fixed['SupExp'] = True
	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('Eb',1.)
		m.minos('p0',1.)
		m.minos('p1',1.)
		m.minos('p2',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['Eb']
		fit_err[i] = m.errors['Eb']
	    if i == 1:
		pfinal[i] = m.values['p0']
		fit_err[i] = m.errors['p0']
	    if i == 2:
		pfinal[i] = m.values['p1']
		fit_err[i] = m.errors['p1']
	    if i == 3:
		pfinal[i] = m.values['p2']
		fit_err[i] = m.errors['p2']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Fit Function provided by Fermi tools to optical depth with fixed Eb ----------------------- #
    def MinuitFitTau_Fermi_Eb(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function fitted to optical depth
	Function given by Fermi tools:
	tau(E) = (E - Eb) * p0 + p1 * ln(E/Eb) + p2* ln(E/Eb)**2
	Energies in MeV
    
	pinit[0] = p0
	pinit[1] = p1
	pinit[2] = p2
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 3
	fitfunc = errfunc_tau_fermi_Eb

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(p0,p1,p2):
	    params = p0,p1,p2
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['p0'] = 0.
	    m.values['p1'] = 1.
	    m.values['p2'] = 0.5
	else:
	    m.values['p0'], m.values['p1'], m.values['p2'] = pinit
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['p0'] = limits[i]
		if i == 1:
		    m.limits['p1'] = limits[i]
		if i == 2:
		    m.limits['p2'] = limits[i]
	else:
	    m.limits['p0'] = (-3.e-1,10.)	# set them all positive, so that no upturn in spectra possible
	    m.limits['p1'] = (-3.e-1,10.)
	    m.limits['p2'] = (-3.e-1,10.)
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	#m.fixed['SupExp'] = True
	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('p0',1.)
		m.minos('p1',1.)
		m.minos('p2',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['p0']
		fit_err[i] = m.errors['p0']
	    if i == 1:
		pfinal[i] = m.values['p1']
		fit_err[i] = m.errors['p1']
	    if i == 2:
		pfinal[i] = m.values['p2']
		fit_err[i] = m.errors['p2']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - Fit Function to fit supernova type II optical light curve --------------------------------- #
    def MinuitFitSN(self,x,y,s,pinit=[],limits=(),fixed=[], full_output=False):
	"""Function to fit super nova type II lightcurve
	according to Cowen et al. 2009, Eq. (1) + Eq. (2)
    
	pinit[0] = a1
	pinit[1] = a2
	pinit[2] = a3
	pinit[3] = t0

	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 4
	fitfunc = errfunc_sn

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(p0,p1,p2,t0):
	    params = p0,p1,p2,t0
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['p0'] = 1.
	    m.values['p1'] = 1.
	    m.values['p2'] = 1.
	    m.values['t0'] = 1.
	else:
	    m.values['p0'], m.values['p1'], m.values['p2'],m.values['t0'] = pinit

	if not len(fixed):		# fix certain parameters?
	    m.fixed['p0'] = False
	    m.fixed['p1'] = False
	    m.fixed['p2'] = False
	    m.fixed['t0'] = False
	else:
	    m.fixed['p0'], m.fixed['p1'], m.fixed['p2'],m.fixed['t0'] = fixed

	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['p0'] = limits[i]
		if i == 1:
		    m.limits['p1'] = limits[i]
		if i == 2:
		    m.limits['p2'] = limits[i]
		if i == 3:
		    m.limits['t0'] = limits[i]
	else:
	    m.limits['p0'] = (-1.e-10,1e10)
	    m.limits['p1'] = (-1.e-10,1e10)
	    m.limits['p2'] = (-1.e-10,1e10)
	    m.limits['t0'] = (1.e-10,1e10)
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('p0',1.)
		m.minos('p1',1.)
		m.minos('p2',1.)
		m.minos('t0',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    if i == 0:
		pfinal[i] = m.values['p0']
		fit_err[i] = m.errors['p0']
	    if i == 1:
		pfinal[i] = m.values['p1']
		fit_err[i] = m.errors['p1']
	    if i == 2:
		pfinal[i] = m.values['p2']
		fit_err[i] = m.errors['p2']
	    if i == 3:
		pfinal[i] = m.values['t0']
		fit_err[i] = m.errors['t0']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# -- Careful Fit Routine ---------------------------------------------------------------------- #
    def MinuitCarefulFit(self,x,y,s,init_func=0,max_func=4,pinit=[],limits=(),
    n_start = 3, n_end = -1, n_step = 1,pbound = 0.05):
	"""
	Careful Fitting with Minuit:
	Fit data pointwise, if fit becomes bad (p < 0.05) try function with more parameters,
	up to the function with index max_func, starting with Index init_func. 
	Indices are:
	    0 MinuitFitPL
	    1 MinuitFitBPL
	    2 MinuitFitDBPL
	    3 MinuitFitEBPL
	    4 MinuitFitEDBPL
	"""
	if n_end < 0:
	    n_end = len(x) + 1
	#pfinal = pinit
	pfinal = []
	ifunc = init_func

	fitfunc = self.MinuitFitPL,\
		  self.MinuitFitBPL,\
		  self.MinuitFitDBPL,\
		  self.MinuitFitEBPL,\
		  self.MinuitFitEDBPL
	for n in range(n_start, n_end,n_step):
	    x_s = x[0:n]
	    fit_stat,pfinal,fit_err = fitfunc[ifunc](
		x_s,y[0:n]*x_s**2.,s[0:n]*x_s**2.,
		pinit = pfinal
		)
	    if fit_stat[2] <= pbound and ifunc < max_func:
		ifunc += 1
		if ifunc == 1:
		    pinit = pfinal
		    pfinal = np.zeros((4,))
		    pfinal[0], pfinal[1] = pinit
		    pfinal[-1] = 1./x[n-1]
		    if n >= len(x):
			pfinal[2] = (np.log(y[-2]) - np.log(y[-1])) \
			    / (np.log(x[-2]) - np.log(x[-1]))
		    else:
			pfinal[2] = (np.log(y[n-1]) - np.log(y[n+1])) \
			    / (np.log(x[n-1]) - np.log(x[n+1]))
		if ifunc == 2:
		    pinit_bpl = pfinal
		    pfinal = np.zeros((6,))
		    pfinal[0:3],pfinal[4] = pinit_bpl[0:3],pinit_bpl[3]
		    pfinal[-1] = 1./x[n-1]
		    if n >= len(x):
			pfinal[3] = (np.log(y[-2]) - np.log(y[-1])) \
			    / (np.log(x[-2]) - np.log(x[-1]))
		    else:
			pfinal[3] = (np.log(y[n-1]) - np.log(y[n+1])) \
			    / (np.log(x[n-1]) - np.log(x[n+1]))
		if ifunc == 3:
		    pinit_dbpl = pfinal
		    pfinal = np.zeros((5,))
		    pfinal[0:4] = pinit_bpl
		    if n >= len(x):
			pfinal[4] = 1./x[-1]
		    else:
			pfinal[4] = 1./x[n]
		if ifunc == 4:
		    pinit = pfinal
		    pfinal = np.zeros((7,))
		    pfinal[0:6] = pinit_dbpl
		    if n >= len(x):
			pfinal[6] = 1./x[-1]
		    else:
			pfinal[6] = 1./x[n]

		fit_stat,pfinal,fit_err= fitfunc[ifunc](
		    x_s,y[0:n]*x_s**2.,s[0:n]*x_s**2.,
		    pinit = pfinal
		    )
	# Convert ifunc to iloop
	if ifunc:
	    if ifunc == 1:
		iloop = 2
	    elif ifunc == 2:
		iloop = 3
	    elif ifunc == 3:
		iloop = 6
	    elif ifunc == 4:
		iloop = 7
	    else:
		iloop = ifunc
	return fit_stat,pfinal,fit_err,iloop,ifunc
# -- Careful Fit Routines ---------------------------------------------------------------------- #
    def MinuitCarefulFitPL(self,x,y,s,pinit=[],limits=(),
    n_start = 3, n_end = -1, n_step = 1):
	return MinuitCarefulFit(self,x,y,s,0,0,pinit,limits,
	n_start, n_end, n_step)

    def MinuitCarefulFitBPL(self,x,y,s,pinit=[],limits=(),
    n_start = 3, n_end = -1, n_step = 1):
	return MinuitCarefulFit(self,x,y,s,0,1,pinit,limits,
	n_start, n_end, n_step)

    def MinuitCarefulFitEBPL(self,x,y,s,pinit=[],limits=(),
    n_start = 3, n_end = -1, n_step = 1):
	return MinuitCarefulFit(self,x,y,s,0,2,pinit,limits,
	n_start, n_end, n_step)

    def MinuitCarefulFitDBPL(self,x,y,s,pinit=[],limits=(),
    n_start = 3, n_end = -1, n_step = 1):
	return MinuitCarefulFit(self,x,y,s,0,3,pinit,limits,
	n_start, n_end, n_step)

    def MinuitCarefulFitDEBPL(self,x,y,s,pinit=[],limits=(),
    n_start = 3, n_end = -1, n_step = 1):
	return MinuitCarefulFit(self,x,y,s,0,4,pinit,limits,
	n_start, n_end, n_step)
