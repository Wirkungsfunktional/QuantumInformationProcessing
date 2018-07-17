import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF
import numpy as np
from matplotlib import pyplot as plt


def half_classic():
    n = 0
    N = 10000

    list = QIF.create_random_ensemble_ginibre(N)
    """
    q = 3/5
    a = np.sqrt(3/4)
    phi1 = a*QIF.q00 + np.sqrt(1 - a**2) * QIF.q11
    phi2 = a*QIF.q01 + np.sqrt(1 - a**2) * QIF.q10
    rho = q * QIF.density_matrix(phi1) + (1 - q)*QIF.density_matrix(phi2)
    """
    pur = []
    pur2 = []

    for rho in list:
        if (QIF.check_density_matrix_half_classical(rho)):
            n += 1
            pur2.append(QIF.purity(rho))
        else:
            pur.append(QIF.purity(rho))

    print(np.mean(pur))
    print(np.max(pur))
    print(np.min(pur2))
    print(np.mean(pur2))
    print(n/N)
