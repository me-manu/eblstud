# Auxilliary functions to calculate bin boundary energies
import numpy as np

def calc_bin_bounds(X):
    """
    calculate bin boundaries for array x assuming that x values lie at logarithmic bin center

    Parameters
    ----------
    X:	n-dim array with logarithmic center values

    Returns
    -------
    (n+1) dim array with bin boundaries
    """
    bin_bounds = np.zeros(X.shape[0] + 1)
    for i,x in enumerate(X[:-1]):
	bin_bounds[i + 1] = np.sqrt(x * X[i + 1])
    bin_bounds[0]	= X[0] **2. / bin_bounds[1]
    bin_bounds[-1]	= X[-1]**2. / bin_bounds[-2]
    return bin_bounds
