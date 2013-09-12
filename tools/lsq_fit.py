"""
Module Containing tools for fitting with least squares scipy routine.

History of changes:
Version 1.0
- Created 7th December 2010
- contains fit for power law (+exp cut off / pile up)
- contains fit for broken power law (+exp cut off / pile up)
- functions to calculate initial conditions 01/07/2011

"""

__version__ = 1.0
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

import numpy as np
from scipy.optimize import leastsq
from scipy.integrate import quad
from eblstud.stats import misc
import scipy.special

gammainc = scipy.special.gammainc
gamma = scipy.special.gamma
pvalue = lambda dof, chisq: 1. - gammainc(.5 * dof, .5 * chisq)

#supexp = 10.
supexp = 1.
cutoff = 21.55
f = 4. # Smoothing parameter
#f2= 4. # Smoothing parameter
#f = 6. # Smoothing parameter
f2= 4. # Smoothing parameter

x_break = 1e3

# Fitting Functions #############################################
fitfunc_pl = lambda p, x: p[0]*(np.power(x,p[1]))
errfunc_pl = lambda p, x, y, err: (fitfunc_pl(p, x)-y) / err

fitfunc_epl = lambda p, x: p[0]*(np.power(x,p[1]))*np.exp(np.power(x*p[2],supexp))
errfunc_epl = lambda p, x, y, err: (fitfunc_epl(p, x)-y) / err

bpl_in = lambda x,xb: 1. + np.power(x*np.abs(xb),f)
bpl_in2 = lambda x,xb: 1. + np.power(x*np.abs(xb),f2)
fitfunc_bpl = lambda p, x: p[0]*(np.power(x,p[1]))*np.power(bpl_in(x,p[3]),(p[2] - p[1])/f)
errfunc_bpl = lambda p, x, y, err: (fitfunc_bpl(p, x)-y) / err

fitfunc_ebpl = lambda p, x: p[0]*(np.power(x,p[1]))*np.power(bpl_in(x,p[3]),(p[2] - p[1])/f)*np.exp(x*p[4])
errfunc_ebpl = lambda p, x, y, err: (fitfunc_ebpl(p, x)-y) / err

fitfunc_dbpl = lambda p, x: p[0]*(np.power(x,p[1]))*np.power(bpl_in(x,p[4]),(p[2] - p[1])/f)*np.power(bpl_in2(x,p[5]),(p[3] - p[2])/f2)
errfunc_dbpl = lambda p, x, y, err: (fitfunc_dbpl(p, x)-y) / err

fitfunc_edbpl = lambda p, x: p[0]*(np.power(x,p[1]))*np.power(bpl_in(x,p[4]),(p[2] - p[1])/f)*np.power(bpl_in2(x,p[5]),(p[3] - p[2])/f2)*np.exp(np.power(x*p[6],supexp))
errfunc_edbpl = lambda p, x, y, err: (fitfunc_edbpl(p, x)-y) / err

fitfunc_edbpl_fs = lambda p, x: p[0]*(np.power(x,p[1]))*np.power(bpl_in(x,p[4]),(p[2] - p[1])/f)*np.power(bpl_in2(x,p[5]),(p[3] - p[2])/f2)*np.exp(np.power(x*p[6],p[7]))
errfunc_edbpl_fs= lambda p, x, y, err: (fitfunc_edbpl_fs(p, x)-y) / err

fitfunc_sebpl_fs = lambda p, x: p[0]*(np.power(x,p[1]))*np.power(bpl_in(x,p[3]),(p[2] - p[1])/f)*np.exp(np.power(x*p[4],p[5]))
errfunc_sebpl_fs = lambda p, x, y, err: (fitfunc_sebpl_fs(p, x)-y) / err

fitfunc_sebpl = lambda p, x: p[0]*(np.power(x,p[1]))*np.power(bpl_in(x,p[3]),(p[2] - p[1])/f)*np.exp(np.power(x*p[4],supexp))
errfunc_sebpl = lambda p, x, y, err: (fitfunc_sebpl(p, x)-y) / err

fitfunc_scebpl = lambda p, x: p[0]*(np.power(x,p[1]))*np.power(bpl_in(x,p[3]),(p[2] - p[1])/f)*np.exp(np.power(x/cutoff,supexp))
errfunc_scebpl = lambda p, x, y, err: (fitfunc_scebpl(p, x)-y) / err

fitfunc_logpar = lambda p,x: p[0]*np.power(x, p[1] + p[2] * np.log(x) )
errfunc_logpar = lambda p,x,y, err: (fitfunc_logpar(p ,x) - y) /err

fitfunc_elogpar = lambda p,x: p[0]*np.power(x,p[1] + p[2] * np.log(x))*np.exp(np.power(x*p[3],supexp))
errfunc_elogpar = lambda p,x,y, err: (fitfunc_elogpar(p ,x) - y) /err

fitfunc_gaussian= lambda p,x: p[0]*np.exp(-0.5 * ((x - p[1]) * p[2])**2.)
errfunc_gaussian= lambda p,x,y, err: (fitfunc_gaussian(p ,x) - y) /err

fitfunc_tau_fermi = lambda p,x: (x - p[0]) * p[1] + p[2] * np.log(x/p[0]) + p[3] * np.log(x/p[0])**2.
errfunc_tau_fermi = lambda p,x,y, err: (fitfunc_tau_fermi(p ,x) - y) /err

fitfunc_tau_fermi_Eb = lambda p,x: (x - x_break) * p[0] + p[1] * np.log(x/x_break) + p[2] * np.log(x/x_break)**2.
errfunc_tau_fermi_Eb = lambda p,x,y, err: (fitfunc_tau_fermi_Eb(p ,x) - y) /err

fitfunc_sn = lambda p,x: p[0]*p[0] * (x-p[3])**1.6 / (np.exp(p[1] * np.sqrt(x-p[3])) - 1.) + p[2] * (x-p[3])*(x-p[3])
errfunc_sn = lambda p,x,y, err: (fitfunc_sn(p ,x) - y) /err

# Jacobians #####################################################
def jacobian_pl(params,xval,yval,yerr):
    norm,ind = params
    J = np.zeros((2,len(xval)))
    J[0:,] = np.power(xval,ind) / yerr
    J[1:,] = np.log(xval)*norm*np.power(xval,ind) / yerr
    return J

def jacobian_epl(params,xval,yval,yerr):
    norm,ind,cut = params
    J = np.zeros((3,len(xval)))
    J[0:,] = np.power(xval,ind)*np.exp(np.power(xval*cut,supexp)) / yerr
    J[1:,] = np.log(xval)*norm*np.power(xval,ind)*np.exp(np.power(xval*cut,supexp))/ yerr
    J[2:,] = norm*np.power(xval,ind+1.)*np.exp(np.power(xval*cut,supexp))*supexp* \
    	np.power(xval * cut, supexp - 1.) / yerr
    return J

def jacobian_bpl(params,xval,yval,yerr):
    norm,ind1,ind2,bre =  params
    bre = np.abs(bre)
    J = np.zeros((4,len(xval)))
    if bre < 0.:# or 1./bre > xval[-1]:
	J[:,:] = 1e100
	return J
    J[0:,] = fitfunc_bpl(params,xval)/norm/yerr
    J[1:,] = fitfunc_bpl(params,xval)*(np.log(xval) - np.log(bpl_in(xval,bre))/f)/yerr
    J[2:,] = fitfunc_bpl(params,xval)*np.log(bpl_in(xval,bre))/f/yerr
    J[3:,] = (ind2 - ind1)*norm*np.power(xval,ind1 + 1.)*np.power(bre*xval,f - 1.)*np.power(bpl_in(xval,bre), (ind2 - ind1)/f - 1.)/yerr
    #J[4:,] = fitfunc_bpl(params,xval) * (ind2 - ind1)/f * (np.power(xval*bre,f)*np.log(xval*bre)/bpl_in(xval,bre) - np.log(bpl_in(xval,bre))/f)/yerr
    return J

def jacobian_scebpl(params,xval,yval,yerr):
    norm,ind1,ind2,bre =  params
    bre = np.abs(bre)
    J = np.zeros((4,len(xval)))
    if bre < 0.:# or 1./bre > xval[-1]:
	J[:,:] = 1e100
	return J
    J[0:,] = fitfunc_scebpl(params,xval)/norm/yerr
    J[1:,] = fitfunc_scebpl(params,xval)*(np.log(xval) - np.log(bpl_in(xval,bre))/f)/yerr
    J[2:,] = fitfunc_scebpl(params,xval)*np.log(bpl_in(xval,bre))/f/yerr
    J[3:,] = (ind2 - ind1)*norm*np.power(xval,ind1 + 1.)*np.power(bre*xval,f - 1.)*np.power(bpl_in(xval,bre), (ind2 - ind1)/f - 1.)/yerr*np.exp(np.power(xval/cutoff,supexp))
    return J

def jacobian_ebpl(params,xval,yval,yerr):
    norm,ind1,ind2,bre,cut =  params
    bre = np.abs(bre)
    J = np.zeros((5,len(xval)))
    if bre < 0.: #or 1./bre > xval[-1]:
	J[:,:] = 1e100
	return J
    J[0:,] = fitfunc_ebpl(params,xval)/norm/yerr
    J[1:,] = fitfunc_ebpl(params,xval)*(np.log(xval) - np.log(bpl_in(xval,bre))/f)/yerr
    J[2:,] = fitfunc_ebpl(params,xval)*np.log(bpl_in(xval,bre))/f/yerr
    J[3:,] = (ind2 - ind1)*norm*np.power(xval,ind1 + 1.)*np.power(bre*xval,f - 1.)*np.power(bpl_in(xval,bre), (ind2 - ind1)/f - 1.)/yerr*np.exp(xval*cut)
    J[4:,] = fitfunc_ebpl(params,xval)*supexp*np.power(cut*xval,supexp)/yerr
    return J

def jacobian_sebpl(params,xval,yval,yerr):
    norm,ind1,ind2,bre,cut=  params
    bre = np.abs(bre)
    J = np.zeros((5,len(xval)))
    if bre < 0.:# or 1./cut < xval[-1] or 1./bre > xval[-1]: 
	J[:,:] = 1e150
	return J
    J[0:,] = fitfunc_sebpl(params,xval)/norm/yerr
    J[1:,] = fitfunc_sebpl(params,xval)*(np.log(xval) - np.log(bpl_in(xval,bre))/f)/yerr
    J[2:,] = fitfunc_sebpl(params,xval)*np.log(bpl_in(xval,bre))/f/yerr
    J[3:,] = (ind2 - ind1)*norm*np.power(xval,ind1 + 1.)*np.power(bre*xval,f - 1.)*np.power(bpl_in(xval,bre), (ind2 - ind1)/f - 1.)/yerr*np.exp(np.power(xval*cut,supexp))
    J[4:,] = fitfunc_sebpl(params,xval)*supexp*np.power(cut*xval,supexp)/yerr/cut
    #J[5:,] = fitfunc_sebpl(params,xval)*np.power(cut*xval,sup)*np.log(cut*xval)/yerr
    return J

def jacobian_dbpl(params,xval,yval,yerr):
    norm,ind1,ind2,ind3,bre1,bre2 =  params
    bre1 = np.abs(bre1)
    bre2 = np.abs(bre2)
    J = np.zeros((6,len(xval)))
#    if bre1 < 0. or bre2 < 0: # or 1./bre1 > xval[-1] or 1./bre2 > xval[-1]:
    #if bre1 < xval[len(xval)/3] or bre2 < x[2*len(xval)/3]:
#	J[:,:] = 1e100
#	return J
    J[0:,] = fitfunc_dbpl(params,xval)/norm/yerr
    J[1:,] = fitfunc_dbpl(params,xval)*(np.log(xval) - np.log(bpl_in(xval,bre1))/f)/yerr
    J[2:,] = (-1.)*fitfunc_dbpl(params,xval)*(np.log(bpl_in(xval,bre1))*f - np.log(bpl_in2(xval,bre2))*f2) /f/f2/yerr
    J[3:,] = fitfunc_dbpl(params,xval)*np.log(bpl_in2(xval,bre2))/f2/yerr
    J[4:,] = (ind2 - ind1)*norm*np.power(xval,ind1 + 1.)*np.power(bre1*xval,f - 1.)*np.power(bpl_in(xval,bre1), (ind2 - ind1)/f - 1.)*np.power(bpl_in2(xval,bre2),(ind3-ind2)/f2)/yerr
    J[5:,] = (ind3 - ind2)*norm*np.power(xval,ind1 + 1.)*np.power(bre2*xval,f2 - 1.)*np.power(bpl_in(xval,bre1), (ind2 - ind1)/f)*np.power(bpl_in2(xval,bre2),(ind3-ind2)/f2 - 1.)/yerr
    return J

def jacobian_edbpl(params,xval,yval,yerr):
    norm,ind1,ind2,ind3,bre1,bre2,cut =  params
    bre1 = np.abs(bre1)
    bre2 = np.abs(bre2)
    J = np.zeros((7,len(xval)))
    #if bre1 < 0. or bre2 < 0:
    #if bre1 < xval[len(xval)/3] or bre2 < xval[2*len(xval)/3]:# or np.abs(cut) < 1./xval[-2]:
#	J[:,:] = 1e100
#	return J
    J[0:,] = fitfunc_edbpl(params,xval)/norm/yerr
    J[1:,] = fitfunc_edbpl(params,xval)*(np.log(xval) - np.log(bpl_in(xval,bre1))/f)/yerr
    J[2:,] = (-1.)*fitfunc_edbpl(params,xval)*(np.log(bpl_in(xval,bre1))*f - np.log(bpl_in2(xval,bre2))*f2 )/f/f2/yerr
    J[3:,] = fitfunc_edbpl(params,xval)*np.log(bpl_in2(xval,bre2))/f2/yerr
    J[4:,] = (ind2 - ind1)*norm*np.power(xval,ind1 + 1.)*np.power(bre1*xval,f - 1.)*np.power(bpl_in(xval,bre1), (ind2 - ind1)/f - 1.)*np.power(bpl_in2(xval,bre2),(ind3-ind2)/f2)/yerr*np.exp(np.power(xval*cut,supexp))
    J[5:,] = (ind3 - ind2)*norm*np.power(xval,ind1 + 1.)*np.power(bre2*xval,f2 - 1.)*np.power(bpl_in(xval,bre1), (ind2 - ind1)/f)*np.power(bpl_in2(xval,bre2),(ind3-ind2)/f2 - 1.)/yerr*np.exp(np.power(xval*cut,supexp))
    J[6:,] = fitfunc_edbpl(params,xval)*xval*np.power(cut*xval,supexp - 1.)*supexp/yerr
    return J

def jacobian_logpar(params,xval,yval,yerr):
    norm,ind,curv=  params
    J = np.zeros((3,len(xval)))
#    if curv > 0.:
#	J[:,:] = 1e100
#	return J
    J[0:,] = fitfunc_logpar(params,xval)/norm/yerr
    J[1:,] = fitfunc_logpar(params,xval)*np.log(xval)/yerr
    J[2:,] = fitfunc_logpar(params,xval)*np.log(xval)**2./yerr
    return J

def jacobian_elogpar(params,xval,yval,yerr):
    norm,ind,curv,cut=  params
    J = np.zeros((4,len(xval)))
#    if curv > 0. or cut < 0.:
#	J[:,:] = 1e100
#	return J
    J[0:,] = fitfunc_elogpar(params,xval)/norm/yerr
    J[1:,] = fitfunc_elogpar(params,xval)*np.log(xval)/yerr
    J[2:,] = fitfunc_elogpar(params,xval)*np.log(xval)**2./yerr
    J[3:,] = supexp* np.exp(np.power(cut*xval,supexp)) * norm *\
	np.power(xval, 1. + ind * curv* np.log(xval) ) * np.power( cut*xval, supexp - 1. ) / yerr
    return J

def jacobian_gaussian(params,xval,yval,yerr):
    norm,mean,var =  params
    J = np.zeros((3,len(xval)))
#    if curv > 0. or cut < 0.:
#	J[:,:] = 1e100
#	return J
    J[0:,] = fitfunc_gaussian(params,xval)/norm/yerr
    J[1:,] = fitfunc_gaussian(params,xval)/yerr * var**2.*(xval - mean*np.ones(len(xval)))
    J[2:,] = fitfunc_gaussian(params,xval)/yerr * (-1.)* var * (xval - mean*np.ones(len(xval)))**2.
    return J

def jacobian_tau_fermi(params,xval,yval,yerr):
    J = np.zeros((4,len(xval)))
#    if curv > 0. or cut < 0.:
#	J[:,:] = 1e100
#	return J
    J[0:,] = -1. * (params[1] + params[2]/params[0] + 2.*params[3]/params[0] * np.log(xval/params[0]))
    J[1:,] = xval - params[0]*np.ones(len(xval))
    J[2:,] = np.log(xval/params[0])
    J[3:,] = np.log(xval/params[0])**2.
    return J
def jacobian_tau_fermi_Eb(params,xval,yval,yerr):
    J = np.zeros((3,len(xval)))
#    if curv > 0. or cut < 0.:
#	J[:,:] = 1e100
#	return J
    J[0:,] = xval - x_break*np.ones(len(xval))
    J[1:,] = np.log(xval/x_break)
    J[2:,] = np.log(xval/x_break)**2.
    return J

def jacobian_sn(params,xval,yval,yerr):
    J = np.zeros((4,len(xval)))
    J[0:,] = 2.*params[0]*(xval - np.ones(len(xval))*params[3]) * params[3]**1.6\
	    /(np.exp(params[1] * np.sqrt(xval - np.ones(len(xval)) * params[3])) - np.ones(len(xval))) /yerr
    J[1:,] = 2.*params[2]*(xval - np.ones(len(xval))*params[3])
    J[2:,] = -params[0]*params[0] * np.exp(params[1] * np.sqrt(xval - np.ones(len(xval)) * params[3])) \
	* xval**2.1/np.power(np.exp(params[1] * np.sqrt(xval - np.ones(len(xval)) * params[3])) - np.ones(len(xval)),2.)/yerr
    J[3:,] = (2. * params[2] * (np.ones(len(xval)) * params[3] - xval) + \
	params[0]*params[0] * (1.6 * (xval - np.ones(len(xval)) * params[3])**0.6 \
	+ np.exp(params[1] * np.sqrt(xval - np.ones(len(xval)) * params[3])) \
	* (-1.6 * (xval - np.ones(len(xval)) * params[3])**0.6 \
	+ 0.5 * params[1] * (xval - np.ones(len(xval)) * params[3])**1.1))\
	/(np.exp(params[1] * np.sqrt(xval - np.ones(len(xval)) * params[3])) - np.ones(len(xval))))\
	/yerr
    return J

# Priors (initial conditions) #####################################################
def prior_norm(x,y):
    """returns flux that corresponds to Energy closest to 1 TeV. x needs to be in TeV

    x,y need to be lists"""

    from eblstud.tools.list_tools import best_index
    return y[best_index(x,1.)[1]]

def prior_pl_ind(x,y):
    """returns prior for power law index by calculating index between y[1] and y[-2] or y[-1]"""

    if len(y) > 3:
	plind = (np.log(y[-2]) - np.log(y[1])) / (np.log(x[-2]) - np.log(x[1]))
    else:
	plind = (np.log(y[-1]) - np.log(y[0])) / (np.log(x[-1]) - np.log(x[0]))
    return plind

def prior_epl_ind(x,y):
    """returns prior for power law index for power law with exp cut-off / pile-up
    by calculating index between y[1] and y[-3] or y[-2]"""

    if len(y) > 4:
	plind = (np.log(y[-3]) - np.log(y[1])) / (np.log(x[-3]) - np.log(x[1]))
    else:
	plind = (np.log(y[-2]) - np.log(y[0])) / (np.log(x[-2]) - np.log(x[0]))
    return plind

def prior_epl_cut(x,y):
    """returns prior for 1/cut off energy for (broken) power law with exp cut-off / pile-up"""
    c = 1./x[-2]
    if y[-1] >= y[-3]:
	return c
    else:
	return (-1)*c

def prior_bpl(x,y):
    """returns prior for broken power law indeces and break energy"""

    b = x[len(x)/2]
    if len(y) > 5:
	plind1 = (np.log(y[len(x)/2-1]) - np.log(y[1])) / (np.log(x[len(x)/2-1]) - np.log(x[1]))
    else:
	plind1 = (np.log(y[len(x)/2]) - np.log(y[1])) / (np.log(x[len(x)/2]) - np.log(x[1]))
    plind2 = (np.log(y[-2]) - np.log(y[len(x)/2])) / (np.log(x[-2]) - np.log(x[len(x)/2]))
    return plind1,plind2,1./b

def prior_ebpl(x,y):
    """returns prior for broken power law indeces and break energy for broken pl with exp cut/pile up"""
    b = x[len(x)/2]
    if len(x) > 6:
	#plind2 = (np.log(y[-3]) - np.log(y[len(x)/2])) / (np.log(x[-3]) - np.log(x[len(x)/2]))
	#b = x[len(x)/2-1]
	plind2 = (np.log(y[-4]) - np.log(y[len(x)/2-1])) / (np.log(x[-4]) - np.log(x[len(x)/2-1]))
	plind1 = (np.log(y[len(x)/2-2]) - np.log(y[1])) / (np.log(x[len(x)/2-2]) - np.log(x[1]))
    else:
	plind1 = (np.log(y[len(x)/2-1]) - np.log(y[1])) / (np.log(x[len(x)/2-1]) - np.log(x[1]))
	plind2 = (np.log(y[-2]) - np.log(y[len(x)/2])) / (np.log(x[-2]) - np.log(x[len(x)/2]))
    return plind1,plind2,1./b

def prior_dbpl(x,y):
    """returns prior for double broken power law indeces and break energy"""

    b1 = x[3*len(x)/4]
    b2 = x[-2]
    plind1 = (np.log(y[3*len(x)/4]) - np.log(y[1])) / (np.log(x[3*len(x)/4]) - np.log(x[1]))
    plind2 = (np.log(y[-2]) - np.log(y[3*len(x)/4-1])) / (np.log(x[-2]) - np.log(x[3*len(x)/4-1]))
    plind3 = (np.log(y[-1]) - np.log(y[-2])) / (np.log(x[-1]) - np.log(x[-2]))
    #plind3 = (np.log(y[-1]) - np.log(y[-2])) / (np.log(x[-1]) - np.log(x[-2]))
    return plind1,plind2,plind3,1./b1,1./b2


def prior_logpar(x,y):
    """ return prior for log paraboliv fit"""
    from eblstud.tools.list_tools import best_index

    # Norm the function not to 1 TeV but to x[ len(x) / 3]
    norm = x[ len(x) / 3]
    x = np.array(map(lambda x: x/norm, x))

    index = best_index(x,1.)[1]
    if len(x) == index + 1:
	plind = (np.log(y[index]) - np.log(y[index-1])) / (np.log(x[index]) - np.log(x[index-1]))
    elif index > 0:
	plind = (np.log(y[index+1]) - np.log(y[index-1])) / (np.log(x[index+1]) - np.log(x[index-1]))
    else:
	plind = (np.log(y[2]) - np.log(y[0])) / (np.log(x[2]) - np.log(x[0]))
    # log Emax of log parabola
    logEmax = np.log(x[0]) - 1.
    curv = (-1.) * np.abs( plind / (2.* logEmax) )
    return plind,curv

def prior_gaussian(x):
    return np.mean(x),np.sqrt(np.var(x))


#################################
def exponent(y):
    """return exponent of basis 10 of y, e.g. y = 1.2e10 than result is 10."""
    if (y - 10**round(np.log10(y))) >= 0:
        return round(np.log10(y))
    else:
	return round(np.log10(y)) - 1.

#################################

def lsq_plfit(x,y,s,pinit=[],full_output = False):
    """Function to fit Powerlaw to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index
    """

    npar = 2

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1] = prior_pl_ind(x,y)

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)

    out = leastsq(errfunc_pl, pinit,
	       args=(x, y, s), Dfun=jacobian_pl, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)

    else:
	fit_err = np.zeros((npar,))

    chisq = sum([errfunc_pl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_eplfit(x,y,s,pinit=[],full_output = False):
    """Function to fit Powerlaw with exp. cut-off to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index
    pinit[2] = 1 / Cut off Energy
    """
    npar = 3

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1
    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1] = prior_epl_ind(x,y)
	pinit[2] = prior_epl_cut(x,y)


    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)

    out = leastsq(errfunc_epl, pinit,
	       args=(x, y, s), Dfun=jacobian_epl, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))

#    fit_err[2] /=np.power(pfinal[2],2.)

    chisq = sum([errfunc_epl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
#    pfinal[2] = 1./pfinal[2]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_bplfit(x,y,s,pinit=[],full_output = False):
    """Function to fit broken Powerlaw to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index 1
    pinit[2] = pl index 2
    pinit[3] = 1 / Break Energy
    """

    npar = 4

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2],pinit[3] = prior_bpl(x,y)

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)


    out = leastsq(errfunc_bpl, pinit,
	       args=(x, y, s), Dfun=jacobian_bpl, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))

    #fit_err[3] /=np.power(pfinal[3],2.)

    chisq = sum([errfunc_bpl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_scebplfit(x,y,s,pinit=[],full_output = False):
    """Function to fit broken Powerlaw with fixed super exponential cut off 
    to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index 1
    pinit[2] = pl index 2
    pinit[3] = 1 / Break Energy
    """

    npar = 4

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2],pinit[3] = prior_bpl(x,y)
    elif len(pinit) > npar:
	pinit = pinit[0:npar]

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)


    out = leastsq(errfunc_scebpl, pinit,
	       args=(x, y, s), Dfun=jacobian_scebpl, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))

    #fit_err[3] /=np.power(pfinal[3],2.)

    chisq = sum([errfunc_scebpl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_ebplfit(x,y,s,pinit=[],full_output=False):
    """Function to fit broken Powerlaw with exponential cut-off/pile-up to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index 1
    pinit[2] = pl index 2
    pinit[3] = 1 / Break Energy
    pinit[4] = 1 / Cut Off Energy
    """
    npar = 5
    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1
    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((5,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2],pinit[3] = prior_ebpl(x,y)
	pinit[4] = prior_epl_cut(x,y)
	#pinit[4] = 1./x[-1] 

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)

    out = leastsq(errfunc_ebpl, pinit,
	       args=(x, y, s), Dfun=jacobian_ebpl, full_output=1, col_deriv=1
	       )
    pfinal = out[0]

    covar = out[1]
    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))

    #fit_err[3] /=np.power(pfinal[3],2.)
    #fit_err[4] /=np.power(pfinal[4],2.)

    chisq = sum([errfunc_ebpl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]
    #pfinal[4] = 1./pfinal[4]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_sebplfit(x,y,s,pinit=[],full_output=False):
    """Function to fit broken Powerlaw with super
    exponential cut-off/pile-up to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index 1
    pinit[2] = pl index 2
    pinit[3] = 1 / Break Energy
    pinit[4] = 1 / Cut Off Energy
    pinit[5] = Super Exponential Parameter <- Set to 10 and constant
    """
    npar = 5
    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1
    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2],pinit[3] = prior_ebpl(x,y)
	#pinit[4] = prior_epl_cut(x,y)
    pinit[4] = 1./x[-1]
	#pinit[5] = 10.

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)

    out = leastsq(errfunc_sebpl, pinit,
	       args=(x, y, s), Dfun=jacobian_sebpl, full_output=1, col_deriv=1,
	       ftol = 1e-7, xtol = 1e-7
	       )
    pfinal = out[0]

    covar = out[1]
    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))

    #fit_err[3] /=np.power(pfinal[3],2.)
    #fit_err[4] /=np.power(pfinal[4],2.)

    chisq = sum([errfunc_sebpl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]
    #pfinal[4] = 1./pfinal[4]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_dbplfit(x,y,s,pinit=[],full_output = False):
    """Function to fit broken Powerlaw to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index 1
    pinit[2] = pl index 2
    pinit[3] = pl index 3
    pinit[4] = 1 / Break Energy 1
    pinit[5] = 1 / Break Energy 2
    """

    npar = 6

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2],pinit[3],pinit[4],pinit[5] = prior_dbpl(x,y)

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)


    out = leastsq(errfunc_dbpl, pinit,
	       args=(x, y, s), Dfun=jacobian_dbpl, full_output=1, col_deriv=1,
	       ftol = 1e-7, xtol = 1e-7
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))

    #fit_err[3] /=np.power(pfinal[3],2.)

    chisq = sum([errfunc_dbpl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_edbplfit(x,y,s,pinit=[],full_output = False):
    """Function to fit a double broken Powerlaw with exp. pile up/ cut off
    to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index 1
    pinit[2] = pl index 2
    pinit[3] = pl index 3
    pinit[4] = 1 / Break Energy 1
    pinit[5] = 1 / Break Energy 2
    pinit[6] = 1 / Cut Off Energy 
    """

    npar = 7

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2],pinit[3],pinit[4],pinit[5] = prior_dbpl(x,y)
#	pinit[6] = prior_epl_cut(x,y)
	pinit[6] = 1./x[-1]

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)


    out = leastsq(errfunc_edbpl, pinit,
	       args=(x, y, s), Dfun=jacobian_edbpl, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))

    #fit_err[3] /=np.power(pfinal[3],2.)

    chisq = sum([errfunc_edbpl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_sebplfit_prior(x,y,s,pinit=[],full_output=False):
    """Function to fit broken Powerlaw with super
    exponential cut-off/pile-up to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl index 1
    pinit[2] = pl index 2
    pinit[3] = 1 / Break Energy
    pinit[4] = 1 / Cut Off Energy
    pinit[5] = Super Exponential Parameter <- Set to 10 and constant
    """
    npar = 5
    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1
    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2],pinit[3] = prior_ebpl(x,y)
	#pinit[4] = prior_epl_cut(x,y)
    pinit[4] = 1./x[-1]
	#pinit[5] = 10.

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)

    out = leastsq(errfunc_sebpl, pinit,
	       args=(x, y, s), Dfun=jacobian_sebpl, full_output=1, col_deriv=1
	       )
    pfinal = out[0]

    covar = out[1]
    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))

    #fit_err[3] /=np.power(pfinal[3],2.)
    #fit_err[4] /=np.power(pfinal[4],2.)

    chisq = sum([errfunc_sebpl(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]
    #pfinal[4] = 1./pfinal[4]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_logpar(x,y,s,pinit=[],full_output = False):
    """Function to fit a log parabola
    to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl 
    pinit[2] = cuvature
    """

    npar = 3

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2] = prior_logpar(x,y)

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)

    # Norm the function not to 1 TeV but to x[ len(x) / 3]
    norm = x[ len(x) / 3]
    x = np.array(map(lambda x: x/norm, x))

    out = leastsq(errfunc_logpar, pinit,
	       args=(x, y, s), Dfun=jacobian_logpar, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))


    chisq = sum([errfunc_logpar(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_elogpar(x,y,s,pinit=[],full_output = False):
    """Function to fit a log parabola
    to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = pl 
    pinit[2] = cuvature
    pinit[3] = 1. / Cut off Energy
    """

    npar = 4

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)


    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_norm(x,y)
	pinit[1],pinit[2] = prior_logpar(x,y)
	pinit[3] = 1./x[-2]

    exp =(-1.)*exponent(y[0])
    y = map(lambda x: x*np.power(10.,exp),y)
    s = map(lambda x: x*np.power(10.,exp),s)
    pinit[0] *= np.power(10.,exp)

    # Norm the function not to 1 TeV but to x[ len(x) / 3]
    norm = x[ len(x) / 3]
    x = np.array(map(lambda x: x/norm, x))

    out = leastsq(errfunc_elogpar, pinit,
	       args=(x, y, s), Dfun=jacobian_elogpar, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
	fit_err[0] *=np.power(10.,-1.*exp)
	covar[:,0] *=np.power(10.,-1.*exp)
	covar[0,:] *=np.power(10.,-1.*exp)
    else:
	fit_err = np.zeros((npar,))


    chisq = sum([errfunc_elogpar(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    pfinal[0] *= np.power(10.,-1.*exp)
    #pfinal[3] = 1./pfinal[3]

    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_gaussian(x,y,s,pinit=[],full_output = False):
    """Function to fit a Gaussian
    to data using scipy.leastsq
    
    pinit[0] = norm
    pinit[1] = mean
    pinit[2] = 1. / sqrt(Variance)
    """

    npar = 3

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)


    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = 1.
	pinit[1],pinit[2] = prior_gaussian(x)


    out = leastsq(errfunc_gaussian, pinit,
	       args=(x, y, s), Dfun=jacobian_gaussian, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
    else:
	fit_err = np.zeros((npar,))

    chisq = sum([errfunc_gaussian(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_tau_fermi(x,y,s,pinit=[],full_output = False):
    """Function fitted to optical depth
    Function given by Fermi tools:
    tau(E) = (E - Eb) * p0 + p1 * ln(E/Eb) + p2* ln(E/Eb)**2
    Energies in TeV
    
    pinit[0] = Eb
    pinit[1] = p0
    pinit[2] = p1
    pinit[3] = p2
    """

    npar = 4

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)


    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = 0.01		# Eb is 10 GeV
	pinit[1] = 1. 
	pinit[2] = 1.
	pinit[3] = 0.5


    out = leastsq(errfunc_tau_fermi, pinit,
	       args=(x, y, s), Dfun=jacobian_tau_fermi, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
    else:
	fit_err = np.zeros((npar,))

    chisq = sum([errfunc_gaussian(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_tau_fermi_Eb(x,y,s,pinit=[],full_output = False):
    """Function fitted to optical depth
    Function given by Fermi tools:
    tau(E) = (E - Eb) * p0 + p1 * ln(E/Eb) + p2* ln(E/Eb)**2
    Energies in TeV
    Eb is 10 GeV
    
    pinit[0] = p0
    pinit[1] = p1
    pinit[2] = p2
    """

    npar = 3

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)


    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = 0.01
	pinit[1] = 1.
	pinit[2] = 0.5


    out = leastsq(errfunc_tau_fermi_Eb, pinit,
	       args=(x, y, s), Dfun=jacobian_tau_fermi_Eb, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
    else:
	fit_err = np.zeros((npar,))

    chisq = sum([errfunc_gaussian(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err
    
def lsq_sn(x,y,s,pinit=[],full_output = False):
    """Function to fit super nova type II lightcurve
    according to Cowen et al. 2009, Eq. (1) + Eq. (2)
    
    pinit[0] = a1
    pinit[1] = a2
    pinit[2] = a3
    pinit[3] = t0
    """

    npar = 4

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)


    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = 1.
	pinit[1] = 1.
	pinit[2] = 1.
	pinit[3] = 1.


    out = leastsq(errfunc_sn, pinit,
	       args=(x, y, s), Dfun=jacobian_sn, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
    else:
	fit_err = np.zeros((npar,))

    chisq = sum([errfunc_gaussian(pfinal,x[i],y[i],s[i])**2 for i in range(len(x))])
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

# --- Propagation of uncertainty for fit functions ---------------------------------#
def PL_unc(pfinal, covar, E):
    """
    Returns uncertainty on PL flux with final parameters pfinal = (Norm, index)
    and covariance covar for energy E
    """
    if np.isscalar(E):
	E = np.array([E])
    ones = np.ones(len(E))
    result = (covar[0,0]/pfinal[0])**2. * ones + (covar[1,1] * np.log(E))**2 + 2.*covar[1,0]/pfinal[0] * np.log(E)
    return result * fitfunc_pl(pfinal,E)**2.

def LP_unc(pfinal, covar, E):
    """
    Returns uncertainty on log par flux with final parameters pfinal = (Norm, index)
    and covariance covar for energy E
    """
    if np.isscalar(E):
	E = np.array([E])
    ones = np.ones(len(E))
    result = (covar[0,0]/pfinal[0])**2. * ones + (covar[1,1] * np.log(E))**2 + (covar[2,2] * np.log(E)**2.)**2.
    result += 2. * np.log(E) * (ones * covar[0,1] / pfinal[0] + covar[0,2] / pfinal[0] * np.log(E) + covar[1,2] * np.log(E)**2.)
    return result * fitfunc_logpar(pfinal,E)**2.
