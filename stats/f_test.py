import numpy as np
from scipy.stats import f
from scipy.stats import t

def f_test(chi1,df1,chi2,df2,red_chi = True):
    """
    F Test to compare hypothesis 1 against hypothesis 2.
    Returns the significance that hypothesis 1 is more probable than hypothesis 2,
    i.e. if close to one, hypothesis one is preferred.

    Parameters
    ----------
    chi1: n-dim array / scalar, chi^2 value of first hypothesis test 
    df1: n-dim array / scalar, degrees of freedom of first hypothesis test 
    chi2: n-dim array / scalar, chi^2 value of second hypothesis test 
    df2: n-dim array / scalar, degrees of freedom of second hypothesis test 
    red_chi: if True, F-test is calculated for reduced chi values

    Returns
    -------
    p-value of F-test (float)
    """

#    if chi1/df1 > chi2/df2:
#	prob = 2. * f.cdf(chi1/df1, chi2/df2, df1, df2)
#    else:
#	prob = 2. * f.cdf(chi2/df2, chi1/df1, df2, df1)
    if red_chi:
	fval = (chi1/df1) / (chi2/df2)
    else:
	fval = chi1 / chi2
    prob = 2. * f.cdf((chi1/df1) / (chi2/df2), df1, df2)
    if prob > 1.: 
	return 2. - prob
    else:
	return prob

def f_test_var(data1,data2):
    """
    F Test to test hypothesis if two samples have different variances.
    H0: samples have same variances (p-value close to one).

    Parameters
    ----------
    data1: n,1 - dim array with data
    data2: n,1 - dim array with data

    Returns
    -------
    p-value of F test

    Notes
    -----
    See 3rd Edition of Numerical recipes chapter 14.2.2, p.730
    """
    var1, var2 = np.var(data1,ddof = 1),np.var(data2,ddof = 1)	# compute variance
    df1, df2, = len(data1) - 1, len(data2) - 1		# compute degrees of freedom
    if var1 > var2:
	prob = 2. * f.cdf(var1/var2,df1,df2)
    else:
	prob = 2. * f.cdf(var2/var1,df2,df1)
    if prob > 1.:
	return 2. - prob
    else:
	return prob

def t_test1(data1,data2):
    """
    Compute t test for two samples with same variance to test if they have same mean
    H0: samples have same means (p-value close to one).

    Parameters
    ----------
    data1: n,1 - dim array with data
    data2: n,1 - dim array with data

    Returns
    -------
    p-value of t test, the t value itself and the degrees of freedom

    Notes
    -----
    See 3rd Edition of Numerical recipes chapter 14.2.1, p.727
    """
    if not isinstance(data1,np.ndarray):
	data1 = np.array(data1)
    if not isinstance(data2,np.ndarray):
	data2 = np.array(data2)

    N1, N2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    # Eq. 14.2.1
    sD = np.sqrt( (np.sum( (data1 - np.ones(N1) * mean1) ** 2.) + np.sum( (data2 - np.ones(N2) * mean2) ** 2.)) / (N1 + N2 - 2.) * (1./N1 + 1./N2))
    T = (mean1 - mean2) / sD
    return t.cdf(T, N1 + N2 - 2),T,N1 + N2 - 2

def t_test2(data1,data2):
    """
    Compute t test for two samples with significantly different variances (use f_test_var) to test if they have same mean
    H0: samples have same means (p-value close to one).

    Parameters
    ----------
    data1: n,1 - dim array with data
    data2: n,1 - dim array with data

    Returns
    -------
    p-value of t test, the t value itself and the degrees of freedom

    Notes
    -----
    See 3rd Edition of Numerical recipes chapter 14.2.1, p.728
    """
    N1, N2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    var1, var2= np.var(data1,ddof = 1), np.var(data2,ddof = 1)

    T = (mean1 - mean2) / np.sqrt(var1/N1 + var2/N2)	# Eq. 14.2.3
    df = (var1/N1 + var2/N2)**2. / ( (var1/N1)**2./(N1 - 1) + (var2/N2)**2./(N2 - 1))
    return t.cdf(T, df), T, df
