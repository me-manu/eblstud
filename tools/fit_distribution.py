# - Imports ------------------------- #
import numpy as np
import iminuit 
import warnings,logging
from scipy.stats import poisson
from scipy.stats import chi2, lognorm, ncx2
from scipy.stats import norm
from scipy.stats import kstest
# ----------------------------------- #

class FitDistribution(object):
    def __init__(self, **kwargs):
	"""
	kwargs
	------
	distr:		str, name of distribution. Available options are: chi2 (default), lognorm, ncx2 
	ksmode:		str, mode for kstest, see scipy.stats.kstest for more details (default: asymp)
	full_output:	bool, if True, errors will be estimated additionally with minos, covariance matrix will also be returned
	print_level:	0,1, level of verbosity, defualt = 0 means nothing is printed
	int_steps:	float, initial step width, multiply with initial values of errors, default = 0.1
	strategy:	0 = fast, 1 = default (default), 2 = thorough
	tol:		float, required tolerance of fit = 0.001*tol*errordef, default = 1.
	errordef:	float, errordef, 1 (default) for chi^2, 0.5 for log-likelihood
	ncall:		int, number of maximum calls, default = 1000
	pedantic:	bool, if true (default), give all warnings
	"""
	kwargs.setdefault('print_level',0)		# no output
	kwargs.setdefault('int_steps',0.1)		# Initial step width, multiply with initial values in m.errors
	kwargs.setdefault('strategy',1)			# 0 = fast, 1 = default, 2 = thorough
	kwargs.setdefault('tol',1.)			# Tolerance of fit = 0.001*tol*UP
	kwargs.setdefault('errordef',1.)		# 1 for chi^2, 0.5 for log-likelihood
	kwargs.setdefault('ncall',1000.)		# number of maximum calls
	kwargs.setdefault('pedantic',True)		# Give all warnings
	kwargs.setdefault('distr','chi2')		
	kwargs.setdefault('ksmode','asymp')		

	self.conf = kwargs
	self.__dict__.update(kwargs)

	if self.distr == 'chi2':
	    self.func = self.chi2
	if self.distr == 'lognorm':
	    self.func = self.lognorm
	if self.distr == 'ncx2':
	    self.func = self.ncx2

	return

    def lognorm(self, s, mean):
	return self.__fit_func(s,mean)

    def chi2(self, df):
	return self.__fit_func(df)

    def ncx2(self, df, nc):
	return self.__fit_func(df, nc)

    def __fit_func(self,*args):
	return kstest(self.data,self.distr,args = args, mode = self.ksmode)[0]

    def fit_dist(self,data,pinit = {}, error = {}, fix = {}, limit = {}):

	self.data	 = data

	pars = iminuit.describe(self.func)

	if not len(fix.keys()):
	    for k in pars:
		fix[k] = 0
	if not len(pinit.keys()):
	    for k in pars:
		pinit[k] = 1.
	if not len(error.keys()):
	    for k in pars:
		error[k] = pinit[k] * self.int_steps
	if not len(limit.keys()):
	    for k in pars:
		limit[k] = [pinit[k] / 1e5, pinit[k] * 1e5]

	fitarg = {}
	for k in pars:
	    fitarg[k] = pinit[k]
	    fitarg['error_{0:s}'.format(k)] = error[k]
	    fitarg['fix_{0:s}'.format(k)] = fix[k]
	    fitarg['limit_{0:s}'.format(k)] = limit[k]

	m		= iminuit.Minuit(self.func, 
			    errordef = self.errordef, pedantic = self.pedantic, print_level = self.print_level,
			    **fitarg
			    )
	m.tol		= self.tol
	m.strategy	= self.strategy

	m.migrad(ncall = self.ncall)

	return m.fval, m.values, m.errors
