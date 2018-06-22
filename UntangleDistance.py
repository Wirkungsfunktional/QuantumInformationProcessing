import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF
import numpy as np
from matplotlib import pyplot as plt






def untangle_dist_hist():
    """Create an ensemble of density matrices and determine the histogram of
    the untangle_dist for all entangled states."""
    N = 5000
    list = QIF.create_random_ensemble_ginibre(N)
    dp = 0.01
    dist = []
    for rho in list:
        p = QIF.untangle_dist(rho, dp)
        if p > 0:
            dist.append(p)

    bb = np.linspace(0, 1, 100)
    plt.hist(dist, bins=bb, normed = 1)
    plt.show()


untangle_dist_hist()
