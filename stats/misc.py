from scipy.integrate import quad
from scipy.special import gamma
import numpy as np
import math
import warnings
import subprocess
from scipy.special import ndtri

def chi2pdf(z,n):
    result = z**(n/2. - 1)*np.exp(-1.*z/2.)
    result /= 2**(n/2.)*gamma(n/2.)
    return result

def chi2pval(x,n):
    if x == 0.:
	return 0.0
    result = quad(chi2pdf,0.0,x,args=n)
    return 1. - result[0]


poisson = lambda l,k : l**k / math.factorial(k) * math.exp(-l)

def sum_poisson(mean, start, stop):
    """
    Calculate sum over poisson sitribution from start to stop (not included)
    with mean
    """
    start = int(start)
    stop = int(stop)
    if stop >= 170:
	warnings.warn("{0} too large for conversion of factorial to float\nreturning 1.".format(stop), RuntimeWarning)
	return 1.
    result = 0.
    for k in range(start, stop):
	result += poisson(mean, k)
    return result

pvalue = lambda dof, chisq: 1. - gammainc(.5 * dof, .5 * chisq)

def Npvalue(dof,chisq,prec = 20):
    """
    numerically compute pvalue using mathematica
    for chisq, and dof degrees of freedom to precision prec (standard is 20, max is 100)
    mathematica needs to be installed and kernel is called with math
    """

    if not isinstance(dof,int):
	dof = int(dof)
	warnings.warn("dof = {0} is not an integer, converted to int.".format(dof), RuntimeWarning)

    str = "Print[1-N[N[Gamma[{0}/2,0,{1:.100f}],{2}]/N[Gamma[{0}/2],{2}],{2}]] Exit[]".format(dof,chisq/2.,prec)

    try:
	p = subprocess.Popen(["math", "-noprompt", "-run", str], stdout=subprocess.PIPE)
    except OSError:
	warnings.warn("mathematica is not installed! returning -1", RuntimeWarning)
	return -1

    out, err = p.communicate()
    pre = float(out.split()[0].split('`')[0])
    if out.find('^') > 0:
	exp = float(out.split()[0].split('`')[1].split('^')[-1])
    else:
	exp = 0.


    if not pre:
	warnings.warn("Npvalue: Precision not sufficient!", RuntimeWarning)
	return pre

    return pre * 10.** exp

def Npvalue_inv(dof,chisq,prec = 20):
    """
    Same as Npvalue but computes 1 - Npvalue
    """

    if not isinstance(dof,int):
	dof = int(dof)
	warnings.warn("dof = {0} is not an integer, converted to int.".format(dof), RuntimeWarning)

    str = "Print[N[Gamma[{0}/2,0,{1:.100f}],{2}]/N[Gamma[{0}/2],{2}]] Exit[]".format(dof,chisq/2.,prec)

    try:
	p = subprocess.Popen(["math", "-noprompt", "-run", str], stdout=subprocess.PIPE)
    except OSError:
	warnings.warn("mathematica is not installed! returning -1", RuntimeWarning)
	return -1

    out, err = p.communicate()
    pre = out.split()[0].split('`')[0]
    if pre.find('*') >= 0:
	pre = out.split()[0].split('*')[0]

    pre = float(pre)
    if out.find('^') > 0:
	if out.find('`') >= 0:
	    exp = float(out.split()[0].split('`')[1].split('^')[-1])
	if out.find('*') >= 0:
	    exp = float(out.split()[0].split('*')[1].split('^')[-1])
    else:
	exp = 0.


    if not pre:
	warnings.warn("Npvalue: Precision not sufficient!", RuntimeWarning)
	return pre

    return pre * 10.** exp
