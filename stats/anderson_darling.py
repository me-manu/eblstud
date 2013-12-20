"""
Anderson-Darling Test for normality
"""

import numpy as np
from scipy.stats import norm

def And_Darl_Stat(x):
    """
    Return Anderson Darling Test Statistic A^2
    Mean and variance are unknown
    See http://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test
    """

    cdf = norm.cdf
    x = np.array(x)
    n,ones = float(len(x)), np.ones(len(x))
    Y = (x - np.mean(x) * ones) / (np.sqrt(np.var(x)) * ones)
    Y = np.sort(Y)
    ni = np.array(range(int(n)))
    Asq = -n - np.sum( (2.*ni - ones)*np.log(cdf(Y)) 
	+ (2.*(n*ones - ni) + ones)*np.log(ones - cdf(Y))) / n
    return Asq*(1. + 4./n -25/n**2)
