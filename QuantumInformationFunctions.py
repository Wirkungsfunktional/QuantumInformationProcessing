import numpy as np
import scipy.linalg as sl
from matplotlib import pyplot as plt



# Qubit states in computational basis
q0 = np.array([1, 0])
q1 = np.array([0, 1])

q00 = np.outer(q0, q0).flatten()
q01 = np.outer(q0, q1).flatten()
q10 = np.outer(q1, q0).flatten()
q11 = np.outer(q1, q1).flatten()

psi_m = (q01 - q10) / np.sqrt(2)
psi_p = (q01 + q10) / np.sqrt(2)
phi_m = (q00 - q11) / np.sqrt(2)
phi_p = (q00 + q11) / np.sqrt(2)

sigma_y_4d = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])


# Computational Functions
def shannon_1bit(x):
    """Compute the classical entropy of the bit system, that '0' occure with
    probability x and '1' with (1 - x). It is zero at x = 0 or x = 1 and maximal
    at x = 0.5 which means 1."""
    if x == 1 or x == 0:
        return 0 # The computation by formular will fail due to the limit case
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

def check_majorisation_of_vectors(x, y):
    """Check whether x ~ y: \sum^d x_i <= \sum^d y_i for x_i, y_i descending
    ordered for all d."""
    assert len(x) == len(y), "Not equal length."
    ret = 1
    sx = 0
    sy = 0
    x[::-1].sort()      # Sorting in descending order
    y[::-1].sort()
    for i in range(len(x)):
        sx += x[i]
        sy += y[i]
        ret = (ret and (sx <= sy))
    return ret

def check_majorisation_of_matrices(A, B):
    """Check whether A ~ B, defined by \Lambda(A) ~ \Lambda(B), with \Lambda(X)
    the vector of all eigenvalues of X."""
    ew_A, ev = np.linalg.eig(A)
    ew_B, ev = np.linalg.eig(B)
    return check_majorisation_of_vectors(ew_A, ew_B)
