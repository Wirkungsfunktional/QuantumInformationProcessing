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



# Qubit states in computational basis
q0 = np.array([0, 1])
q1 = np.array([1, 0])

q00 = np.outer(q0, q0).flatten()
q01 = np.outer(q0, q1).flatten()
q10 = np.outer(q1, q0).flatten()
q11 = np.outer(q1, q1).flatten()

psi_m = (q01 - q10) / np.sqrt(2)
psi_p = (q01 + np.sqrt(29)*q10) / np.sqrt(30)
phi_m = (q00 - q11) / np.sqrt(2)
phi_p = (q00 + q11) / np.sqrt(2)

sigma_y_4d = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])


# Computational Functions
def shannon_1bit(x):
    """Compute the classical entropy of the bit system, that '0' occure with
    probability x and '1' with (1 - x). It is zero at x = 0 or x = 1 and maximal
    at x = 0.5 which means 1."""
    return -x * np.log2(x) - (1-x) * np.log2(1 - x)

def entanglement_2qbit(c):
    """Compute the Entanglement of Formation according to Woot1998 of an 2-Qubit
    State. It takes the Concurrency (@see concurrency(rho) ) as input."""
    return shannon_1bit( (1 + np.sqrt(1 - c**2) )/2.0 )

def density_matrix(state):
    """Compute the outer product of $\ket{state}\bra{state}$ of an Vector state."""
    return np.outer(state, state)

def werner_state(p, state):
    """Compute the werner state, which is the superposition of an entangled
    state (state) and the totally mixed state, which is the Identity operator."""
    return p*density_matrix(state) + (1 - p)*np.eye(4)/4.0

def concurrency(rho):
    """Compute the Concurrency of a desity matrix according to Woot1998."""
    rho_tilde = np.dot(sigma_y_4d, np.dot(rho, sigma_y_4d))
    R = np.dot(rho, rho_tilde)
    ew, ev = np.linalg.eig(R)
    ew = np.sort(np.sqrt(ew))
    return np.max([0, ew[3] - ew[0] - ew[1] - ew[2]])

def fidelity(rho1, rho2):
    """Compute the Fidelity of two desity operators. The Fidelity is a measure,
    how far apart two matrices are. Here the square of the Trace will be
    returned. Sometimes the root of this is called by fidelity."""
    try:
        rho1_sqrt = sl.sqrtm(rho1)
        R = sl.sqrtm(np.dot( rho1_sqrt , np.dot(rho2, rho1_sqrt) ))
    except:
        print("Computation failed, return 0 as fidelity.")
        return 0
    return np.trace(R)**2


# Main functions
def plot_entanglement():
    n = 500
    p = np.linspace(0, 1, n)
    entanglement = np.zeros(n)
    conc = np.zeros(n)
    dist = np.zeros(n)
    for i, pp in enumerate(p):
        w = werner_state(pp, psi_m)
        dist[i] = fidelity(w, np.dot(sigma_y_4d, np.dot(w, sigma_y_4d)))
        conc[i] = concurrency(w)
        entanglement[i] = entanglement_2qbit(conc[i])

    plt.plot(p, dist, label='Fidelity of rho and rho_tilde')
    plt.plot(p, entanglement, label="Entanglement of Formation")
    plt.plot(p, conc, label="Concurrency")
    plt.xlabel("p")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    plot_entanglement()
