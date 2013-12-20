
"""
This module contains functions to compute the Kolmogorov-Smirnov test
between two distributions using numpy arrays.

History if changes:
Version 1.0
- Created (09/22/2010)

"""

__version__ = 1.0
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

import numpy as np

def cdf(x) :
    """
    Computes the CDF of a list or an array and returns a sorted np array with tuples x, CDF(x)
    """
    if not isinstance(x,np.ndarray):
	x = np.array(x)
    x = np.sort(x)
    return np.array([x,(np.array(range(len(x)))+1.)/len(x)])

def cdf_arr(x) :
    """
    Computes the CDF of (n x m)-dim array along 1st (m) axis

    Parameters
    ----------
    x: (n x m)-dim array

    Returns
    -------
    CDF: (n x m)-dim array, for every axis n, the CDF(x[n,:])
    """
    x = np.sort(x)	# sort x along last axis
    result = np.zeros(x.shape)

    return np.array([x,(np.array(range(len(x)))+1.)/len(x)])

def KSdist(z):
    """Computes the p value Q_KS of the Kolmogorov-Smirnov Test.
    See Numerical Recipes Third Ed. Eq. 6.14.56/57 on p. 334

    """
    if np.isscalar(z):
	z = np.array([z])
	return_scalar = True
    else:
	return_scalar = False
    if np.any(z < 0.):
	raise ValueError("bad z in KSdist!")
    result = np.zeros(z.shape[0])
    result[z < 1.18] = y = -np.log(np.exp(-1.23370055013616983/(z*z)))
    if np.any(z < 1.18):
	y = np.exp(-1.23370055013616983/(z[z < 1.18]*z[z < 1.18]))
	result[z < 1.18] = 2.25675833419102515*np.sqrt(-np.log(y))
	result[z < 1.18] *= (y + np.power(y,9.) + np.power(y,25.) + np.power(y,49.))
	result[z < 1.18] = np.ones(z[z < 1.18].shape[0]) - result[z < 1.18]
    if np.any(z >= 1.18):
	x = np.exp(-2.*z[z >= 1.18]*z[z >= 1.18])
	result[z >= 1.18] = 2.*(x - np.power(x,4.) + np.power(x,9.))
    if return_scalar:
	return result[0]
    else:
	return result

from scipy.special import ndtri
def KStest(l1, l2):
    """ 
    Evaluate the Kolmogorov-Smirnov Test for two sets of data points l1 and l2.
    Function returns maximal Distance between the CDFs and the corresponding Q_KS value
    See Numerical Recipes 3rd Edition, p.738
    """
    if not isinstance(l1,np.ndarray):
	l1 = np.array(l1)
    if not isinstance(l2,np.ndarray):
	l2 = np.array(l2)

    cdf1 = cdf(l1)
    cdf2 = cdf(l2)
    D = 0.
    i,j = 0,0
    fn1,fn2 = 0.,0.
    while(i < len(l1) and j < len(l2)):
	d1 = cdf1[0][i] 
	d2 = cdf2[0][j] 
	if d1 <= d2:
	    fn1 = cdf1[1][i]
	    i += 1
	if d2 <= d1:
	    fn2 = cdf2[1][j]
	    j += 1
	d=np.abs(fn2-fn1)
	if d > D:
	    D = d
    Neff = float(len(l1))*float(len(l2)) / float( len(l1) + len(l2))
    result = KSdist( (np.sqrt(Neff) + 0.12 + 0.11/np.sqrt(Neff)) * D )
    return D,result,ndtri(1. - result)

def KStest2(l1, func):
    """ 
    Evaluate the Kolmogorov-Smirnov Test between the data set l1 and the CDF func,
    where func = CDF(x), and 0. <= CDF(x) <= 1. for all x.
    Function returns maximal Distance between the CDFs and the corresponding Q_KS value.

    Parameters:
    -----------
    l1 = (n x m)-dim array 
    func = function of normed CDF, must be able to handle (n x m)-dim arrays

    Returns:
    --------
    (3 x n)-dim tuple with:
	D = maximum distance of KS test (n-dim array)
	result = signficance of KS test (n-dim array)
	result in sigma = significance in sigma (one sided confidence level, n-dim array)

    Notes
    -----
    if n = 1, floats are returned in tuple
    See Numerical Recipes 3rd Edition, p.737


    """
    if not isinstance(l1,np.ndarray):
	l1 = np.array(l1)
    if len(l1.shape) == 1:
	l1 = np.array([l1])
	ret_float = True
    else:
	ret_float = False

    x = np.sort(l1)	# sort l1 along last axis
    cdf_data = (np.mgrid[0:x.shape[0],0:x.shape[1]][1] + np.ones(x.shape)) / x.shape[1]	# compute cdf(x), same for all n rows, is (n x m) array
    cdf_func = func(x)	# ( n x m )-dim array

    D = np.max(np.abs(cdf_data - cdf_func), axis = 1)	# compute maximum distance over axis 1. D is n-dim array

    Neff	= np.sqrt(x.shape[1]) + 0.12 + 0.11/np.sqrt(x.shape[1])
    QKS		= KSdist( Neff  * D )
    sig		= -1. * ndtri(QKS)
    if ret_float:
	return D[0],QKS[0],sig[0]
    else:
	return D,QKS,sig

def plot_KStest1(l1, l2, filename = 'None', xlabel = 'None'):
    """ 
    Evaluate the Kolmogorov-Smirnov Test between the data set l1 and l2,
    where func = CDF(x), and 0. <= CDF(x) <= 1. for all x.
    Function plots CDFs of distributions.

    Parameters:
    -----------
    l1 = list or n-array 
    func = function of normed CDF
    filename = name of pdf file where output is saved to. If none, output is shown.

    Returns:
    --------
    Nothing

    Notes
    -----
    See Numerical Recipes 3rd Edition, p.737
    """
    import matplotlib.pyplot as plt
    from math import floor,ceil

    D,QKS,sigma = KStest(l1, l2)	# do KS test, results are written in plot
    l1_cdf = cdf(l1)	# return (l1 in sorted order, cdf(l1))
    l2_cdf = cdf(l2)	# return (l1 in sorted order, cdf(l1))
    xerr1 = np.zeros(l1_cdf[0].shape[0])	# compute x errorbars
    xerr2 = np.zeros(l2_cdf[0].shape[0])	# compute x errorbars

    for i,x in enumerate(l1_cdf[0]):
	if i == l1_cdf[0].shape[0] - 1:
	    xerr1[i] = 0.
	else:
	    xerr1[i] = l1_cdf[0][i + 1] - x
    for i,x in enumerate(l2_cdf[0]):
	if i == l2_cdf[0].shape[0] - 1:
	    xerr2[i] = 0.
	else:
	    xerr2[i] = l2_cdf[0][i + 1] - x

    # the plot
    fig = plt.figure() 
    plt.errorbar(l1_cdf[0], l1_cdf[1], xerr = np.array( [ np.zeros(l1_cdf[0].shape[0]), xerr1]),
	ls = 'None',
	marker = 'o',
	color = '0.'
	)
    plt.errorbar(l2_cdf[0], l2_cdf[1], xerr = np.array( [ np.zeros(l2_cdf[0].shape[0]), xerr2]),
	ls = 'None',
	marker = 's',
	color = 'red'
	)
    logQKS = floor(np.log10(QKS))

    plt.annotate('$D = {0:.2f}$\n$Q_\mathrm{{KS}} = {1:.2f}\,\\times10^{{-}}{{}}^{2:n} = {3:.2f}\,\sigma$'.format(D,QKS / 10.**logQKS,-logQKS,sigma), xy = (0.05,0.9), xycoords = 'axes fraction', backgroundcolor = 'white')

    if not xlabel == 'None':
	plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid(True)

    plt.axis([np.min([l1_cdf[0][0],l2_cdf[0][0]]) * 1.1,np.max([l1_cdf[0][-1] + xerr1[-1],l2_cdf[0][-1] + xerr2[-1]]) * 1.1,0.,1.1])
    if filename == 'None':
	plt.show()
    else:
	plt.savefig('{0}'.format(filename), format = 'pdf')
	plt.savefig('{0}'.format(filename.split('.pdf')[0] + '.eps'), format = 'eps')
    return

def plot_KStest2(l1, func, filename = 'None', xlabel = 'None'):
    """ 
    Evaluate the Kolmogorov-Smirnov Test between the data set l1 and the CDF func,
    where func = CDF(x), and 0. <= CDF(x) <= 1. for all x.
    Function plots CDFs of distributions.

    Parameters:
    -----------
    l1 = list or n-array 
    func = function of normed CDF
    filename = name of pdf file where output is saved to. If none, output is shown.

    Returns:
    --------
    Nothing

    Notes
    -----
    See Numerical Recipes 3rd Edition, p.737
    """
    import matplotlib.pyplot as plt

    D,QKS,sigma = KStest2(l1, func)	# do KS test, results are written in plot
    l1_cdf = cdf(l1)	# return (l1 in sorted order, cdf(l1))
    xerr = np.zeros(l1_cdf[0].shape[0])	# compute x errorbars
    for i,x in enumerate(l1_cdf[0]):
	if i == l1_cdf[0].shape[0] - 1:
	    xerr[i] = 0.
	else:
	    xerr[i] = l1_cdf[0][i + 1] - x

    # the plot
    fig = plt.figure() 
    plt.errorbar(l1_cdf[0], l1_cdf[1], xerr = np.array( [ np.zeros(l1_cdf[0].shape[0]), xerr]),
	ls = 'None',
	marker = 'o',
	color = '0.'
	)

    x_plot = np.linspace(l1_cdf[0][0],l1_cdf[0][-1],100)
    plt.plot(x_plot,func(x_plot), ls = '-', lw = '2.', color = 'red')
    plt.annotate('$D = {0:.2f}$\n$Q_\mathrm{{KS}} = {1:.2e}$'.format(D,QKS), xy = (0.05,0.9), xycoords = 'axes fraction', backgroundcolor = 'white')

    if not xlabel == 'None':
	plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid(True)

    plt.axis([l1_cdf[0][0],l1_cdf[0][-1],0.,1.1])
    if filename == 'None':
	plt.show()
    else:
	plt.savefig('{0}'.format(filename), format = 'pdf')
	plt.savefig('{0}'.format(filename.split('.pdf')[0] + '.eps'), format = 'eps')
    return
