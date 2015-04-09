from eblstud.misc.constants import M_E_EV
from numpy import invert,power,log10,zeros,invert, array, sum, pi

# ---- fitting constants for line absorption ----- #
def set_constants(C = None):
    """
    Set the fitting constants for the optical depth for absorption in BLR
    
    kwargs
    ------
    C:     either list of length 6 with fitting constants or None
    """
    if type(C) == list:
        if not len(C) == 6:
            raise ValueError("List must contain exaclty 6 floats.")
    elif C == None:
        C = []
        C.append(-0.34870519)
        C.append(0.50134530)
        C.append(-0.62043951)
        C.append(-0.022359406)
        C.append(0.059176392)
        C.append(0.0099356923)
    C = array(C)
    global A
    A = C
    return 

def line_absorption(EGeV, **kwargs):
    """
    Calculate optical depth due to interaction with absoption lines
    
    Parameters
    ----------
    EGeV:      n-dim array, energy in GeV
    
    kwargs
    ------
    z:         float, source redshift (default: 0)
    Eline_eV:  float, line energy in rest frame in eV
    Nline:     float, column density im cm^-2
    
    Returns
    -------
    n-dim array with optical depth
    
    Notes
    -----
    See Justin's lecture notes and e.g. Poutanen and Stern (2010) for BLR photon energies, 
    http://adsabs.harvard.edu/abs/2010ApJ...717L.118P
    
    """
    kwargs.setdefault('z',0.)
    kwargs.setdefault('Eline_eV',13.6)
    kwargs.setdefault('Nline',1e25)
    x			= EGeV * 1e9 * (1. + kwargs['z']) * kwargs['Eline_eV'] / M_E_EV**2. - 1.
    m			= x > 0.
    x[m]		= log10(x[m])
    x[invert(m)]	= zeros(sum(invert(m)))
    result		= zeros((A.shape[0],x.shape[0]))
    for i,a in enumerate(A):
        result[i][m]	= A[i] * power(x[m],i)

    result		= 10.**sum(result, axis = 0)
    result[invert(m)]	= zeros(sum(invert(m)))
    return result * kwargs['Nline'] * pi * 2.8179e-13 ** 2.
