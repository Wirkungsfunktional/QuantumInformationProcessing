import numpy as np
import scipy.linalg as sl
from matplotlib import pyplot as plt
import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF


__doc__ = """
Program to determine the amount of 2-qbit-states which are entangled. The idea
is to create an ensemble of random states which are uniform with respect to the
purity. It is expected, that the amount of entangled states is 8/33.
It seems, that entanglement with a mentionalble amount occurs at approximatlly
pur = 1/3, but this have to be checked against the werner state. There could be
a set of zero measure of impure entangled states.
"""

def ensemble_test1():
    e = []
    pur = []
    N = 10000
    pur_end = 0.5
    rho_check = np.zeros( (4, 4) ) + 0.j
    for rho in QIF.create_random_ensemble_ginibre(N):
        #QIF.create_random_ensemble_arcsin(4, 2, 5000):
        #create_2qubit_random_density_matrix_ensemble(pur_end, N):
        rho_check += rho
        conc = QIF.concurrency(rho)
        pur.append(QIF.purity(rho).real)
        e.append(QIF.entanglement_2qbit(conc))
    e = np.array(e)
    pur = np.array(pur)
    notentangled = len(e) - np.count_nonzero(e)
    pur_ent = pur[np.nonzero(e)]
    entangled = np.count_nonzero(e)
    n = len(e)
    print(rho_check/n)
    print("Seperable: " + str(notentangled/n))
    print("Entangled: " + str(entangled/n))
    print("Expect: " + str(8.0/33.0))
    bb = np.linspace(0.25, (1 + 3*pur_end)/4.0, N)
    bb = bb - (bb[1] - bb[0])/2.0
    bb = np.linspace(0.25, 1, 100)
    plt.hist(pur, bins=bb, label="all")
    plt.hist(pur_ent, bins=bb, label="entangled")
    plt.legend()
    plt.show()

def __ensemble_test2():
    # deprecated
    """The states which can be produced by this procedure have a purity below
    the level needed to produce a remarkable amount of entanglement."""
    e = []
    pur = []
    N = 1000
    pur_end = 0.5
    rho_check = np.zeros( (4, 4) ) + 0.j
    for rho in QIF.create_2qubit_random_density_matrix_ensemble_by_pratial_trace(pur_end, N):
        rho_check += rho
        conc = QIF.concurrency(rho)
        pur.append(QIF.purity(rho).real)
        e.append(QIF.entanglement_2qbit(conc))
    e = np.array(e)
    pur = np.array(pur)
    notentangled = len(e) - np.count_nonzero(e)
    pur_ent = pur[np.nonzero(e)]
    entangled = np.count_nonzero(e)
    n = len(e)
    print(notentangled/n)
    print(entangled/n)
    print(rho_check/n)
    print("Expect: " + str(8.0/33.0))
    bb = np.linspace(0.25, (1 + 3*pur_end)/4.0, N)
    bb = bb - (bb[1] - bb[0])/2.0
    bb = np.linspace(0.25, 1, 100)
    plt.hist(pur)
    plt.show()

def ensemble_test3(nn = 1):
    """"""
    e = []
    pur = []
    N = 10000
    rho_check = np.zeros( (4, 4) ) + 0.j
    notentangled = 0
    entangled = 0
    n = 0
    plot_opt = 1
    for rho in QIF.create_2qubit_random_density_matrix_ensemble_by_random_unitary(N, nn):
        n += 1
        rho_check += rho
        if plot_opt:
            conc = QIF.concurrency(rho)
            if QIF.entanglement_2qbit(conc) == 0:
                notentangled += 1
            e.append(QIF.entanglement_2qbit(conc))

        pur.append(QIF.purity(rho))
        rho2 = QIF.partial_transpose(rho)
        ev, ew = np.linalg.eig(rho2)
        if (np.min(ev.real) < 0):
            entangled += 1

    print(rho_check/n)
    print("Entangled: " + str(entangled/n))
    print("Seperable: " + str(1 - entangled/n))
    print("Expect: " + str(8.0/33.0))

    if plot_opt:
        e = np.array(e)
        pur = np.array(pur)
        pur_ent = pur[np.nonzero(e)]
        pur_nent = pur[np.where(e == 0)]

        bb = np.linspace(0.25, 1, 50)
        plt.hist(pur, bins=bb)
        plt.hist(pur_ent, bins=bb)
        plt.show()
        plt.hist(pur_ent, bins=bb)
        plt.hist(pur_nent, bins=bb)
        plt.show()


if __name__ == '__main__':
    print(__doc__)
    ensemble_test3()
