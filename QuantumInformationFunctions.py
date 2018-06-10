import numpy as np
import scipy.linalg as sl
from matplotlib import pyplot as plt
from typing import List
import ClassicalInformationFunctions as CIF

__doc__ = """ Collection of functions releated to Quantum Information Theory.
The implementation is only for learning purpose, there is no focus on
performance and optimaization.


"""

#TODO: Partial transpose



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

def analyse_desity_matrix(rho):
    """Perform a list of tools on a given matrix and prints its results."""
    assert check_density_operator(rho), "rho is not a density matrix"
    size_a, size_b = rho.shape
    pur = purity(rho).real
    fid_to_impur = fidelity(rho, np.eye(size_a)/size_a).real
    entropy = von_neuman_entropy(rho)

    print("Purity: " + str(pur))
    print("Fidelity to totally impure state: " + str(fid_to_impur))
    print("Von-Neuman entropy " + str(entropy))

def von_neuman_entropy(rho):
    """Diagonalize the density matrix and calculate the classical entropy for
    this distribution."""
    assert check_density_operator(rho), "rho is not a density matrix"
    ew, ev = np.linalg.eigh(rho)
    return CIF.classical_entropy(np.abs(ew))



def create_2qubit_random_density_matrix_ensemble_by_random_unitary(N, n):
    return [make_pure_random_2qbit_density_matrix_by_unitary(1.0, n) for p in np.linspace(0, 1, N)**(0.5)]


def random_unitary(n):
    H = 10 * np.random.randn(n, n) + 10j*(np.random.randn(n, n))
    Q, R = sl.qr(H)
    return Q

def make_random_1qubit_density_matrix(p, random_purity_flag = 0):
    """Create a random density matrix for 1 qubit. p gives the purity of the 
    matrix and random_purity_flag activate a random purity."""
    n = 20
    r = np.random.rand(3)*n - n/2 # np.random.normal(scale=1.0, size=(15)) #*np.random.rand()
    r = r / np.dot(r, r)**0.5 * p * (1 - random_purity_flag*np.random.rand())
    d = np.zeros((2, 2)) + 0.j

    for i in range(3):
        d += r[i]*sigma[i]/2.0
    return d + np.eye(2)/2.0



def make_pure_random_2qbit_density_matrix_by_unitary(p, n):
    rho = np.zeros( (4, 4) ) + 0.j
    pp = np.random.rand(n)
    pp = pp/np.sum(pp)
    for i in range(n):
        rho += np.kron(make_random_1qubit_density_matrix(p), make_random_1qubit_density_matrix(p))*pp[i]
    U = random_unitary(4)
    #U = np.kron(random_unitary(2),random_unitary(2))
    rho = np.dot(U, np.dot(rho, np.conjugate(np.transpose(U))))
    return rho

def partial_transpose(rho):
    rho[1][0], rho[0][1] = rho[0][1], rho[1][0]
    rho[1][2], rho[0][3] = rho[0][3], rho[1][2]
    rho[2][1], rho[3][0] = rho[3][0], rho[2][1]
    rho[2][3], rho[3][2] = rho[3][2], rho[2][3]
    return rho


def get_index_in_computational_basis(binary_number_string):
    number_string = "0b" + binary_number_string[::-1]
    return int(number_string, 2)

def make_computational_basis_index(int_number, size):
    return ('{0:0' + str(size) + 'b}').format(int_number)[::-1]

def partial_trace(rho_ab):
    size_a, size_b = rho_ab.shape
    assert size_b == size_a, "matrix not symmetric"
    number_of_systems = int(np.log2(size_a))
    reduced_size = 2**(number_of_systems - 1)
    rho_red = np.zeros( (reduced_size, reduced_size) ) + 0.j
    for i in range(size_a):
        for j in range(size_a):
            bin_string_1 = make_computational_basis_index(i, number_of_systems)
            bin_string_2 = make_computational_basis_index(j, number_of_systems)
            if (bin_string_1[0] == bin_string_2[0]):
                rho_red [get_index_in_computational_basis(bin_string_1[1:])] \
                        [get_index_in_computational_basis(bin_string_2[1:])] += \
                    rho_ab[get_index_in_computational_basis(bin_string_1)] \
                        [get_index_in_computational_basis(bin_string_2)]
    return rho_red



def partial_trace2(rho_ab):
    size_a, size_b = rho_ab.shape
    assert size_b % 4 == 0, "not 4 * N size of Matrix"
    n = int(size_b / 2)
    n2 = int(n/2)
    rho_a = np.zeros( (n, n) ) + 0.j
    rho_a[:n2,:n2] = rho_ab[0:n2,0:n2] + rho_ab[n2:2*n2,n2:2*n2]
    rho_a[:n2,n2:] = rho_ab[2*n2:3*n2,0:n2] + rho_ab[3*n2:,n2:2*n2]
    rho_a[n2:,:n2] = np.conjugate(rho_a[:n2,n2:])
    rho_a[n2:,n2:] = rho_ab[2*n2:3*n2,2*n2:3*n2] + rho_ab[3*n2:,3*n2:]
    return rho_a

def create_2qubit_random_density_matrix_ensemble(pur_end, N: int) -> List[np.ndarray]:
    """Create a List of Matrices. The check for positiv definitnes is done by
    using the existence of a cholesky decomposition. Better would be a check of
    the positivity of all minors of the matrix to have positiv semi definit
    check. The random ensemble is not uniform with respect to purity. Impure
    states are prefered."""
    ensemble = []
    n = 0
    p = np.linspace(0, pur_end, N)**0.5
    while n < N:
        rho = make_random_2qubit_density_matrix(p[n])
        if check_density_operator_property_hermiticty(rho):
            if check_density_operator_property_positiv(rho):
                print(n)
                ensemble.append(rho)
                n += 1
    return ensemble

def check_minor_of_matrix(m):
    d = np.linalg.det(m)
    test_flag = np.isclose( d,  0.0) or d > 0.0
    for i in range(len(m)):
        h = np.concatenate( (m[:i], m[i+1:]) )
        h = np.concatenate( (h[:,:i], h[:,i+1:]), axis=1)
        if (len(h) != 0) and test_flag:
            test_flag = test_flag and check_minor_of_matrix(h)
    return test_flag

def create_2qubit_random_density_matrix_ensemble_by_pratial_trace(pur_end, N: int) -> List[np.ndarray]:
    """Create a List of Matrices. The check for positiv definitnes is done by
    using the existence of a cholesky decomposition. Better would be a check of
    the positivity of all minors of the matrix to have positiv semi definit
    check. The random ensemble is not uniform with respect to purity. Impure
    states are prefered."""
    ensemble = []
    n = 0
    while n < N:
        rho = make_random_4qubit_density_matrix()
        if purity(rho) <= 1.0:
            if check_density_operator_property_positiv(rho):
                print(n)
                m = partial_trace(rho)
                m1 = partial_trace(m)
                ensemble.append(m1)
                n += 1
    return ensemble



def make_random_4qubit_density_matrix() -> np.ndarray:
    """Creates a density matrix of a 4 Qubit state. There is no check for the
    positivity of the matrix. To be save preform a density matrix check
    afterwards."""
    n = 20
    r = np.random.normal(scale=1.0, size=(93)) #*np.random.rand()
    r = r / np.dot(r, r)**0.5 * np.random.rand() * 0.25 # * p
    a = r[:3]#np.random.rand(3)*n - int(n/2)
    b = r[3:6]#np.random.rand(3)*n - int(n/2)
    c = r[6:9]
    d = r[9:12]
    m = r[12:].reshape( (3, 3, 3, 3) )#np.random.rand(3, 3)*n - int(n/2)
    rho = np.zeros((16, 16)) + 0.j

    for i in range(3):
        rho += a[i]*np.kron(np.kron(np.kron(sigma[i], Id), Id), Id)
        rho += b[i]*np.kron(np.kron(np.kron(Id, sigma[i]), Id), Id)
        rho += c[i]*np.kron(np.kron(np.kron(Id, Id), sigma[i]), Id)
        rho += d[i]*np.kron(np.kron(np.kron(Id, Id), Id), sigma[i])
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    rho += m[i][j][k][l]*np.kron(np.kron(np.kron(sigma[i], sigma[j]), sigma[k]), sigma[l])
    rho += np.eye(16)/16.0
    return rho


def make_random_2qubit_density_matrix(p) -> np.ndarray:
    """Creates a 4 dimmensional density matrix by using the Hilbert - Schmidt
    representation. The matrix can be pure or impure. Better to produce on
    vector with norm condition."""
    n = 20
    r = np.random.rand(15)*n - n/2 # np.random.normal(scale=1.0, size=(15)) #*np.random.rand()
    r = r / np.dot(r, r)**0.5 * (3/16)**0.5 * p
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
    return p*density_matrix(state) + (1 - p)*np.eye(4)/4.0 + 0.j

def concurrency(rho: np.ndarray) -> float:
    """Compute the Concurrency of a desity matrix according to Woot1998."""
    rho_tilde = np.dot(sigma_y_4d, np.dot(np.transpose(rho), sigma_y_4d))
    R = np.dot(rho, rho_tilde)
    ew, ev = np.linalg.eig(R)
    ew = np.sort(np.sqrt(np.round(ew.real, 5))) # Rounding is a hack to avoid
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

def fidelity_qubit(rho1, rho2):
    return np.trace(np.dot(rho1, rho2)) + 2 * np.sqrt(np.linalg.det(rho1) * np.linalg.det(rho2))


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
    return np.allclose(rho, np.transpose(rho).conjugate())

def check_density_operator_property_positiv(rho):
    try:
        if np.isclose(purity(rho), 1):
            print("rho is pure, positivity check will not perform.")
            # TODO: Find secure way for pure matrices
            return True
        np.linalg.cholesky(rho)
        return True
    except:
        return False

def check_density_operator(rho):
    flag = check_density_operator_property_trace(rho)
    flag = (flag and check_density_operator_property_hermiticty(rho))
    #flag = (flag and check_density_operator_property_positiv(rho))
    return flag

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

def matrix_root2(m):
    """Return the root of a matrix. Due to the fact that the used function:
    scipy.linalg.sqrtm diagonal matrices as singular classify and therefore is
    unusable, there is a check of diagonal."""
    ev, ew = np.linalg.eig(m)
    return np.diag(np.sqrt(ev.real))
