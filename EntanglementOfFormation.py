import numpy as np
import scipy.linalg as sl
from matplotlib import pyplot as plt
import QuantumInformationFunctions as QIF

__doc__ = """This Program should compute the entanglement of formation for an
two-qubit-state, according to
@article{Woot1998,
  author = {William K. Wootters},
  title = {Entanglement of Formation of an Arbitrary State of Two Qubits},
  journal = {Physical Review Letters},
  year = {1998},
}

The entanglement of formation is a measure for mixed states, which extends the
entropy concept for pure states.
\[
 E[\rho_{AB}] = \min_{\{p_i, \ket{\Psi_{AB}^i}\}} \sum p_i E[ \ket{\Psi_{AB}^i}]
\]
with $E[\ket{\Psi_{AB}^i}] = S(\rho{A})$ the entropy of the reduced desity
operator.
The difficulty to evaluate this formular lies on the minimization for all
representations of the density matrix.
"""



# Main functions
def plot_entanglement():
    n = 500
    p = np.linspace(0, 1, n)
    entanglement = np.zeros(n)
    conc = np.zeros(n)
    dist = np.zeros(n)
    for i, pp in enumerate(p):
        w = QIF.werner_state(pp, QIF.psi_m)
        dist[i] = QIF.fidelity(w, np.dot(QIF.sigma_y_4d, np.dot(w, QIF.sigma_y_4d)))
        conc[i] = QIF.concurrency(w)
        entanglement[i] = QIF.entanglement_2qbit(conc[i])

    plt.plot(p, dist, label='Fidelity of rho and rho_tilde')
    plt.plot(p, entanglement, label="Entanglement of Formation")
    plt.plot(p, conc, label="Concurrency")
    plt.xlabel("p")
    plt.legend()
    plt.show()

def rho_AB(a, b):
    a2 = np.abs(a)**2
    ac = np.conjugate(a)
    b2 = np.abs(b)**2
    bc = np.conjugate(b)
    return np.array([   [2/3 * a2,      1/3 * ac*b,     1/3 * ac*b,     0],
                        [1/3 * a*bc,    1/6,            1/6,            1/3 * ac*b],
                        [1/3 * a*bc,    1/6,            1/6,            1/3 * ac*b],
                        [0,             1/3 * a*bc,     1/3 * a*bc,     2/3 * b2 ]])

def copy_entanglement():
    n = 2
    a = np.linspace(0, 1, n)
    c = np.zeros(n)
    E = np.zeros(n)
    for i, aa in enumerate(a):
        r = rho_AB(aa, np.sqrt(1-aa**2))
        c[i] = QIF.concurrency(r)
        E[i] = QIF.entanglement_2qbit(c[i])
    plt.plot(a, c, label="Concurrency")
    plt.plot(a, E, label="Entanglement")
    print(np.mean(c))
    plt.legend()
    plt.show()



if __name__ == '__main__':
    print(__doc__)
    #plot_entanglement()
    copy_entanglement()
