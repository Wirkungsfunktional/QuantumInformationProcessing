import numpy as np
import scipy.linalg as sl
from matplotlib import pyplot as plt

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


def shannon_1bit(x):
    """Compute the classical entropy of the bit system, that '0' occure with
    probability x and '1' with (1 - x). It is zero at x = 0 or x = 1 and maximal
    at x = 0.5 which means 1."""
    return -x * np.log2(x) - (1-x) * np.log2(1 - x)


def entanglement_2qbit(c):
    return shannon_1bit( (1 + np.sqrt(1 - c**2) )/2.0 )


# Qubit states in computational basis
q0 = np.array([0, 1])
q1 = np.array([1, 0])

q00 = np.outer(q0, q0).flatten()
q01 = np.outer(q0, q1).flatten()
q10 = np.outer(q1, q0).flatten()
q11 = np.outer(q1, q1).flatten()

psi_m = (q01 - q10) / np.sqrt(2)
psi_p = (q01 + q10) / np.sqrt(2)
phi_m = (q00 - q11) / np.sqrt(2)
phi_p = (q00 + q11) / np.sqrt(2)

def density_matrix(state):
    return np.outer(state, state)

def werner_state(p, state):
    return p*density_matrix(state) + (1 - p)*np.eye(4)/4.0



def concurrency(rho):
    sigma_y_4d = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
    rho_tilde = np.dot(sigma_y_4d, np.dot(rho, sigma_y_4d))
    #rho_sqrt = sl.sqrtm(rho)
    R_test = np.dot(rho, rho_tilde)
    #R = sl.sqrtm(np.dot(rho_sqrt, np.dot(rho_tilde, rho_sqrt)))
    ew, ev = np.linalg.eig(R_test)
    #ew2, ev2 = np.linalg.eig(R)
    ew = np.sort(np.sqrt(ew))
    return np.max([0, ew[3] - ew[0] - ew[1] - ew[2]])


def plot_entanglement():
    n = 500
    p = np.linspace(0, 1, n)
    entanglement = np.zeros(n)
    conc = np.zeros(n)
    for i, pp in enumerate(p):
        try:
            conc[i] = concurrency(werner_state(pp, psi_p))
            entanglement[i] = entanglement_2qbit(conc[i])
        except:
            pass

    plt.plot(p, entanglement)
    plt.plot(p, conc)
    plt.show()




class QbitRegister():
    def __init__(self, n, state):
        self.number_of_qbits = n
        self.qbits = state
        self.single_qbit_in_comp_basis = [np.array([1, 0]), np.array([0, 1])]

    def get_register_in_comp_basis(self):
        ret = self.single_qbit_in_comp_basis[self.qbits[0]]
        for i in range(1, self.number_of_qbits):
            ret = np.outer(self.single_qbit_in_comp_basis[self.qbits[i]], ret).flatten()
        return ret

    def get_desity_matrix_in_comp_basis(self):
        state_in_comp_basis = self.get_register_in_comp_basis()
        return np.outer(state_in_comp_basis, state_in_comp_basis)

plot_entanglement()
