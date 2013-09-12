"""
Class to calculate EBL attenuation (optical depth) with GSL
C program has to be provided as opt_depth_calc

History of changes:
Version 1.0
- Created 9th January 2011

"""

__version__ = 1.0
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

import numpy as np
from a3p.ebl.attenuation import *
import a3p.tools.bspline
import os

order = 3

def calc_attenuation_GSL(z,E,ebl):
    """
    Calculate optical depth with GSL program for BSpline interpolated ebl (ebl) for redshift z
    and Energy E in TeV. Function returns optical depth.
    """

    file = 'spline2C.dat'
    f = open(file,"w")
    xval = np.linspace(np.log10(0.1),np.log10(1000.),300)
    if ebl.basespline == -1:
	for x in xval:
	    f.write("%e \t %e\n" % (np.power(10.,x),ebl.bspline.eval(x,order)))
    else:
	for x in xval:
	    nuFnu = ebl.bspline.eval_base_spline(x, ebl.basespline, order)
	    f.write("%e \t %e\n" % (np.power(10.,x),nuFnu))
    f.close()
    os.system('./opt_depth_calc %f %f %s > tt' % (z, E, file))
    f = open("tt")
    result = float(f.readlines()[0])
    f.close()
    return result

