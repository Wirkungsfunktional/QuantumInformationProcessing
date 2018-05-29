import numpy as np
import scipy.linalg as sl
from matplotlib import pyplot as plt
from typing import List


__doc__ = """ Collection of functions releated to Quantum Information Theory.
The implementation is only for learning purpose, there is no focus on
performance and optimaization.


"""


#TODO: Partial trace
#TODO: Partial transpose
#TODO: Matrix tensor product



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

sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1.j],[1.j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
sigma = np.array([sigma_x, sigma_y, sigma_z])
Id = np.eye(2)



sigma_y_4d = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])


def create_2qubit_random_density_matrix_ensemble(N: int) -> List[np.ndarray]:
    """Create a List of Matrices. The check for positiv definitnes is done by
    using the existence of a cholesky decomposition. Better would be a check of
    the positivity of all minors of the matrix to have positiv semi definit
    check. The random ensemble is not uniform with respect to purity. Impure
    states are prefered."""
    ensemble = []
    n = 0
    while n < N:
        rho = make_random_2qubit_density_matrix()
        if check_density_operator_property_hermiticty(rho):
            try:
                np.linalg.cholesky(rho)
                ensemble.append(rho)
                n += 1
            except np.linalg.LinAlgError:
                pass
    return ensemble

def check_minor_of_matrix(m):
    test_flag = np.isclose( np.linalg.det(m),  0.0)
    for i in range(len(m)):
        h = np.concatenate( (m[:i], m[i+1:]) )
        h = np.concatenate( (h[:,:i], h[:,i+1:]), axis=1)
        if len(h) != 0:
            test_flag = test_flag and check_minor_of_matrix(h)
    return test_flag




def make_random_2qubit_density_matrix() -> np.ndarray:
    """Creates a 4 dimmensional density matrix by using the Hilbert - Schmidt
    representation. The matrix can be pure or impure. Better to produce on
    vector with norm condition."""
    n = 20
    r = np.random.normal(scale=1.0, size=(15)) #*np.random.rand()
    r = r / np.dot(r, r)**0.5 * (3/16)**0.5 * np.random.rand()
    a = r[:3]#np.random.rand(3)*n - int(n/2)
    b = r[3:6]#np.random.rand(3)*n - int(n/2)
    c = r[6:].reshape( (3, 3) )#np.random.rand(3, 3)*n - int(n/2)
    d = np.zeros((4, 4)) + 0.j

    for i in range(3):
        d += a[i]*np.kron(sigma[i], Id) + b[i]*np.kron(Id, sigma[i])
        for j in range(3):
            d += c[i][j]*np.kron(sigma[i], sigma[j])
    return d + np.eye(4)/4.0


# Computational Functions
def shannon_1bit(x: np.ndarray) -> np.ndarray:
    """Compute the classical entropy of the bit system, that '0' occure with
    probability x and '1' with (1 - x). It is zero at x = 0 or x = 1 and maximal
    at x = 0.5 which means 1."""
    if x == 1 or x == 0:
        return 0 # The computation by formular will fail due to the limit case
    return -x * np.log2(x) - (1-x) * np.log2(1 - x)

def entanglement_2qbit(c: np.ndarray) -> np.ndarray:
    """Compute the Entanglement of Formation according to Woot1998 of an 2-Qubit
    State. It takes the Concurrency (@see concurrency(rho) ) as input."""
    return shannon_1bit( (1 + np.sqrt(1 - c**2) )/2.0 )

def density_matrix(state: np.ndarray) -> np.ndarray:
    """Compute the outer product of $\ket{state}\bra{state}$ of an Vector state."""
    return np.outer(state, state)

def werner_state(p: float, state: np.ndarray) -> np.ndarray:
    """Compute the werner state, which is the superposition of an entangled
    state (state) and the totally mixed state, which is the Identity operator."""
    return p*density_matrix(state) + (1 - p)*np.eye(4)/4.0

def concurrency(rho: np.ndarray) -> float:
    """Compute the Concurrency of a desity matrix according to Woot1998."""
    rho_tilde = np.dot(sigma_y_4d, np.dot(rho, sigma_y_4d))
    R = np.dot(rho, rho_tilde)
    ew, ev = np.linalg.eig(R)
    ew = np.sort(np.sqrt(np.round(ew.real, 3))) # Rounding is a hack to avoid
    # negative allmost zero eigenvalues like -1e-15 in the root
    return np.max([0, ew[3] - ew[0] - ew[1] - ew[2]])

def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute the Fidelity of two desity operators. The Fidelity is a measure,
    how far apart two matrices are. Here the square of the Trace will be
    returned. Sometimes the root of this is called by fidelity."""
    if np.isclose(purity(rho1), 1.0):
        print("rho1 is pure") # TODO: Retrieve State form desity matrix and use
        # diferent formular for this, because the following procedure may fail
        raise ValueError("Pure State")
    if np.isclose(purity(rho2), 1.0):
        print("rho2 is pure")
    try:
        rho1_sqrt = sl.sqrtm(rho1)
        R = matrix_root(np.dot( rho1_sqrt , np.dot(rho2, rho1_sqrt) ))
    except:
        print("Computation failed, return 0 as fidelity.")
        return 0
    return np.trace(R)**2

def check_majorisation_of_vectors(x: np.ndarray, y: np.ndarray) -> bool:
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

def check_majorisation_of_matrices(A: np.ndarray, B: np.ndarray) -> bool:
    """Check whether A ~ B, defined by \Lambda(A) ~ \Lambda(B), with \Lambda(X)
    the vector of all eigenvalues of X."""
    ew_A, ev = np.linalg.eig(A)
    ew_B, ev = np.linalg.eig(B)
    return check_majorisation_of_vectors(ew_A, ew_B)

def purity(rho: np.ndarray) -> float:
    """Gives the purity of a density matrix. By definition a density matrix has
    trace 1. The square of a density matrix is 1 iff it is a pure state
    otherwise it will be below."""
    return np.trace(np.dot(rho, rho))


def qbit_from_bloch_sphere(theta: float, phi: float) -> np.ndarray:
    return np.cos(theta/2.0) * q0 + np.exp(1.j * phi) * np.sin(theta/2.0) * q1

def qbit_density_matrix(n: np.ndarray):
    """Creates a matrix in Hilbert-Schmidt representation. There will be no
    check for the Trace-, Hermiticity- and Positivity property"""
    return Id/2.0 + sigma_x * n[0] + sigma_y * n[1] + sigma_z * n[2]


def check_density_operator_property_trace(rho: np.ndarray) -> bool:
    """Check that the trace of the density matrix is 1."""
    return np.isclose(np.trace(rho), 1)

def check_density_operator_property_hermiticty(rho: np.ndarray) -> bool:
    """Check that the density matrix is hermitian. This means equal to its
    transposed and complex conjugated."""
    return np.array_equal(rho, np.transpose(rho).conjugate())


def partial_trace(rho, dim_a, dim_b, system):
    pass

def kolmogorov_distance(rho1, rho2):
    ew, ev = np.linalg.eig(rho1 - rho2)
    return np.sum(np.abs(ew))/2.0

def trace_norm(A):
    return np.trace(matrix_root(np.dot(np.transpose(A.conjugate()), A)))

def matrix_root(m):
    """Return the root of a matrix. Due to the fact that the used function:
    scipy.linalg.sqrtm diagonal matrices as singular classify and therefore is
    unusable, there is a check of diagonal."""
    if np.count_nonzero(np.round(m - np.diag(m))) == 0: # check for diagonal
        return np.sqrt(m)
    return sl.sqrtm(m)
