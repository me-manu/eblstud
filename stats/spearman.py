"""
This module contains functions to compute the spearman-rank correlation
coefficient of two data sets.

History if changes:
Version 1.0
- Created (11/19/2010)

"""

__version__ = 1.0
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

from scipy.stats.stats import rankdata
from scipy.special import stdtr
from numpy import sqrt

def spearman_rs(l1,l2):
    """Compute Spearman-Rank Correlation Coefficient with corresponding p-Value"""

    if len(l1) == 0 or len(l2) == 0:
	print 'ERROR: LISTS CONTAIN NO ELEMENTS!'
	return -1. 
    elif len(l1) != len(l2):
	print 'ERROR: LISTS HAVE TO HAVE THE SAME LENGTH!'
	return -1. 
    l1 = rankdata(l1)
    l2 = rankdata(l2)
    l1_mean = sum(l1)/len(l1)
    l2_mean = sum(l2)/len(l2)
    sum1 = 0.
    sum2 = 0.
    numerator = 0.
# Compute Spearman rs
    for i in range(0,len(l1)):
	numerator +=(l1[i] - l1_mean)*(l2[i] - l2_mean)
	sum1 += (l1[i] - l1_mean)**2
	sum2 += (l2[i] - l2_mean)**2
    denum = sqrt(sum1)*sqrt(sum2)
    rs = numerator/denum
# Compute Spearman t
    t = len(l1) - 2.
    t /= 1. - rs**2
    t = rs*sqrt(t)
# if t > 0: change sign, since student's t is axis symmetric around zero
    if t>0:
	t_help = (-1.)*t
    else:
	t_help = t
#p = stdtr(len(z)-2.,t_help)
    p = stdtr(len(l1)-2.,t_help)
    return (rs,p)
