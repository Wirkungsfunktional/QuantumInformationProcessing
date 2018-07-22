import numpy as np
import scipy.linalg as sl
from matplotlib import pyplot as plt
from typing import List
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF

__doc__ = """ Collection of functions releated to Quantum Information Theory.
The implementation is only for learning purpose, there is no focus on
performance and optimaization.


"""




# Qubit states in computational basis
q0 = np.array([1, 0])
q1 = np.array([0, 1])

q00 = np.outer(q0, q0).flatten()
q01 = np.outer(q0, q1).flatten()
q10 = np.outer(q1, q0).flatten()
q11 = np.outer(q1, q1).flatten()

q000 = np.kron(q0, np.kron(q0, q0)) + 0.j
q001 = np.kron(q0, np.kron(q0, q1)) + 0.j
q010 = np.kron(q0, np.kron(q1, q0)) + 0.j
q011 = np.kron(q0, np.kron(q1, q1)) + 0.j
q100 = np.kron(q1, np.kron(q0, q0)) + 0.j
q101 = np.kron(q1, np.kron(q0, q1)) + 0.j
q110 = np.kron(q1, np.kron(q1, q0)) + 0.j
q111 = np.kron(q1, np.kron(q1, q1)) + 0.j

GHZ_p = (q000 + q111)/np.sqrt(2)
GHZ_m = (q000 + q111)/np.sqrt(2)

W1 = (q100 + q010 + q001)/np.sqrt(3)
W2 = (q100 + q010 - q001)/np.sqrt(3)
W3 = (q100 - q010 + q001)/np.sqrt(3)
W4 = (q100 - q010 - q001)/np.sqrt(3)
W5 = (-q100 + q010 + q001)/np.sqrt(3)
W5 = (-q100 + q010 - q001)/np.sqrt(3)
W6 = (-q100 - q010 + q001)/np.sqrt(3)
W7 = (-q100 - q010 - q001)/np.sqrt(3)




psi_m = (q01 - q10) / np.sqrt(2)
psi_p = (q01 + q10) / np.sqrt(2)
phi_m = (q00 - q11) / np.sqrt(2)
phi_p = (q00 + q11) / np.sqrt(2)

sigma_x = np.array([[0, 1],[1, 0]]) + 0.j
sigma_y = np.array([[0, -1.j],[1.j, 0]])
sigma_z = np.array([[1, 0],[0, -1]]) + 0.j
sigma = np.array([sigma_x, sigma_y, sigma_z])
Id = np.eye(2)

sigma_y_4d = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])

def func_von_neuman_entropy(rho):
    """Diagonalize the density matrix and calculate the classical entropy for
    this distribution. See [NiCh2010] p. 510 for further information."""
    assert check_density_operator(rho), "rho is not a density matrix"
    ew, ev = np.linalg.eigh(rho)
    return CIF.classical_entropy(np.abs(ew))

def func_quantum_relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    """The quntum analogon to the Kullback-Leibler distance. See [NiCh2010]
    p. 511 for further information."""
    assert check_density_operator(rho), "Rho is not a density operator"
    assert check_density_operator(sigma), "Sigma is not a density operator"
    assert rho.shape == sigma.shape, "Rho has to be of the same dimension as \
    sigma."
    ew_sig, ev_sig = np.linalg.eigh(sigma)
    log_sigma = np.dot(ev_sig, np.dot( np.diag(np.log2(ew_sig)),
                               np.conjugate(ev_sig.T)))
    return (func_von_neuman_entropy(rho) -
            np.trace(np.dot(rho, log_sigma))).real



def check_klein_inequality(rho: np.ndarray, sigma: np.ndarray) -> bool:
    """Check the validity of the klein inequality. See [NiCh2010] p. 511
    for further information."""
    assert check_density_operator(rho), "Rho is not a density operator"
    assert check_density_operator(sigma), "Sigma is not a density operator"
    return func_quantum_relative_entropy(rho, sigma) >= 0.0


def check_joint_entropy_theorem(N: int, n: int) -> bool:
    """Check the validity of the joint entropy theorem. See [NiCh2010] p. 513
    for further information. N is the size of the system B density matrix, n is
    the number of the Set of density matrix of B in the convex combination."""
    assert N >= 1
    assert n >= 1
    base_A = np.eye(n)
    p = np.random.rand(n)
    p /= np.sum(p)
    rhs = CIF.classical_entropy(p)
    rho = np.zeros( (N*n, N*n) ) + 0.j
    for i in range(n):
        rho_B = make_random_density_matrix_from_ginibre(N)
        rho += p[i] * np.kron(density_matrix(base_A[i]), rho_B)
        rhs += p[i]*func_von_neuman_entropy(rho_B)
    return np.isclose(func_von_neuman_entropy(rho), rhs)




def make_state_standard_map(N: int, K: float) -> np.ndarray:
        n = np.arange(0, N, 1)
        m = np.arange(-N/2, N/2, 1)
        V = np.exp(-1.j*N*K/(2*np.pi) * np.cos(2*np.pi*n/N))
        T = np.exp(-1.j*np.pi*m**2/N)
        f = np.fft.ifft(V)
        U = np.zeros((N, N)) * 1.j
        for i in range(N):
            for j in range(N):
                U[i][j] = f[j-i]*T[j]
        return U



def get_schmidt_decomposition(state: np.ndarray, N_a: int, N_b: int):
    # TODO: TestCase
    A = np.zeros( (2**N_a, 2**N_b) ) + 0.j
    base_a = np.eye(2**N_a) + 0.j
    base_b = np.eye(2**N_b) + 0.j
    for i in range(2**N_a):
        for j in range(2**N_b):
            A[i][j] = np.dot(state, np.kron(base_a[i], base_b[j]))
    u, s, vh = np.linalg.svd(A)
    return s, np.dot(np.conjugate(u.T), base_a), np.dot(vh, base_b)

def make_state_from_schmidt_decomposition(sing_val, base_a, base_b):
    # TODO: TestCase
    state = np.zeros(len(base_a)*len(base_b))
    for i in range(len(sing_val)):
        state += sing_val[i] * np.kron(base_a[i], base_b[i])
    return state

def create_base_n_qubit_comp(N: int) -> List[np.ndarray]:
    return np.eye(2**N) + 0.j


def make_n_dim_hadamard_state(N: int) -> np.ndarray:
    """Creates a full superposition of a n-qubit state |0...0>"""
    base = create_base_n_qubit_comp(N)
    state = np.zeros(2**N) + 0.j
    for s in base:
        state += s
    return state/np.sqrt(2**N)

def make_special_state_bound_entangle_2_4():
    pass
def make_special_state_bound_entangle_3_3():
    pass
def make_special_state_half_classical(q = 3/5, a = np.sqrt(3/4)):
    """Make the density matrix for a state which satisfies the entropy
    inequality S_AB <= S_red for one subsystem which is then called as classical
    but the matrix violate the inequality with respect to the other system which
    is therefore quantum. See [ZyHoHoHo2001] for further information."""
    phi1 = a*q00 + np.sqrt(1 - a**2) * q11
    phi2 = a*q01 + np.sqrt(1 - a**2) * q10
    return q *density_matrix(phi1) + (1 - q)*density_matrix(phi2)



def check_density_matrix_half_classical(rho: np.ndarray) -> bool:
    """Check whether the density matrix violate the inequalities
        S_AB <= S_A or
        S_AB <= S_B
    are they both violated the system is completly quantum. If one is
    satisfied than the system is classical with respect to this subsystem."""

    SAB = von_neuman_entropy(rho)
    SA = von_neuman_entropy(partial_trace(rho))
    SB = von_neuman_entropy(partial_trace(information_swap(rho)))
    return (np.sign(SAB - SA) != np.sign(SAB - SB))




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
    ew, ev = np.linalg.eig(rho)
    return CIF.classical_entropy(np.abs(ew))



def create_2qubit_random_density_matrix_ensemble_by_random_unitary(N, n):
    return [make_pure_random_2qbit_density_matrix_by_unitary_partial_trace(1.0, n) for p in np.linspace(0, 1, N)**(0.5)]



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

def make_random_seperable_2qubit_density_matrix():
    rho = np.zeros( (4, 4,) ) + 0.j
    n = np.random.randint(1, high=10)
    p = np.random.normal(scale=1.0, size=(n))
    p = p/np.sum(p)
    for i in range(n):
        rho1 = make_random_1qubit_density_matrix(1, 1)
        rho2 = make_random_1qubit_density_matrix(1, 1)
        rho += p[i]*np.kron(rho1, rho2)
    return rho




def make_pure_random_2qbit_density_matrix_by_unitary(p, n):
    rho = np.zeros( (4, 4) ) + 0.j
    pp = np.random.rand(n)
    pp = pp/np.sum(pp)
    for i in range(n):
        rho += np.kron(make_random_1qubit_density_matrix(p), make_random_1qubit_density_matrix(p))*pp[i]
    U = MF.make_matrix_random_unitary(4, 4)
    rho = np.dot(U, np.dot(rho, np.conjugate(np.transpose(U))))
    return rho

def make_pure_random_2qbit_density_matrix_by_unitary_partial_trace(p, n):
    """Create an ensemble of density matrices by first constructing an Seperable
    state of 4 uncoupled qubits, then applying a random unitary matrix and then
    tracing out 2 qbits."""
    rho = np.zeros( (16, 16) ) + 0.j
    rho +=  np.kron(make_random_1qubit_density_matrix(1),
            np.kron(make_random_1qubit_density_matrix(1),
            np.kron(make_random_1qubit_density_matrix(1),
                    make_random_1qubit_density_matrix(1))))
    U = MF.make_matrix_random_unitary(16, 16)
    rho = np.dot(U, np.dot(rho, np.conjugate(np.transpose(U))))
    rho = partial_trace(partial_trace(rho))
    return rho

def make_pure_random_2qbit_density_matrix_by_unitary_partial_trace2(p, n):
    """Create an ensemble of density matrices by first constructing an Seperable
    state of 8 uncoupled qubits, then applying a random unitary matrix and then
    tracing out to 2 qbits. The state should in principle be closer to the
    totally mixed state than in the procedure with only 4 uncoupled qubits"""
    N = 2**8
    rho = np.zeros( (N, N) ) + 0.j
    rho +=  np.kron(make_random_1qubit_density_matrix(1),
            np.kron(make_random_1qubit_density_matrix(1),
            np.kron(make_random_1qubit_density_matrix(1),
            np.kron(make_random_1qubit_density_matrix(1),
            np.kron(make_random_1qubit_density_matrix(1),
            np.kron(make_random_1qubit_density_matrix(1),
            np.kron(make_random_1qubit_density_matrix(1),
                    make_random_1qubit_density_matrix(1)
                    )))))))
    U = MF.make_matrix_random_unitary(N, N)
    rho = np.dot(U, np.dot(rho, np.conjugate(np.transpose(U))))
    rho = partial_trace(partial_trace(partial_trace(partial_trace(partial_trace(partial_trace(rho))))))
    return rho


def partial_transpose(rho):
    rho = np.copy(rho)
    rho[1][0], rho[0][1] = rho[0][1], rho[1][0]
    rho[1][2], rho[0][3] = rho[0][3], rho[1][2]
    rho[2][1], rho[3][0] = rho[3][0], rho[2][1]
    rho[2][3], rho[3][2] = rho[3][2], rho[2][3]
    return rho

def partial_transpose_a(rho):
    rho = np.copy(rho)
    rho[2][0], rho[0][2] = rho[0][2], rho[2][0]
    rho[1][2], rho[3][0] = rho[3][0], rho[1][2]
    rho[2][1], rho[0][3] = rho[0][3], rho[2][1]
    rho[1][3], rho[3][1] = rho[3][1], rho[1][3]
    return rho

def information_swap(rho):
    rho = np.copy(rho)
    rho[0][1], rho[0][2] = rho[0][2], rho[0][1]
    rho[1][0], rho[2][0] = rho[2][0], rho[1][0]
    rho[1][1], rho[2][2] = rho[2][2], rho[1][1]
    rho[2][1], rho[1][2] = rho[1][2], rho[2][1]
    rho[1][3], rho[2][3] = rho[2][3], rho[1][3]
    rho[3][1], rho[3][2] = rho[3][2], rho[3][1]
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
    rho_a[n2:,:n2] = np.transpose(np.conjugate(rho_a[:n2,n2:]))
    rho_a[n2:,n2:] = rho_ab[2*n2:3*n2,2*n2:3*n2] + rho_ab[3*n2:,3*n2:]
    return rho_a

def partial_trace3(rho, n):
    N, M = rho.shape
    red_rho = np.zeros( (n, n) ) + 0.j
    d = np.eye(n)+ 0.j
    for i in range(n):
        B = np.kron(d, d[i])
        red_rho += np.dot(B, np.dot(rho, B.T))
    return red_rho



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
    assert False, "Deprecated function: make_random_2qubit_density_matrix"
    """Creates a 4 dimmensional density matrix by using the Hilbert - Schmidt
    representation. The matrix can be pure or impure. Better to produce on
    vector with norm condition."""
    n = 20
    r = np.random.rand(15)*n - n/2 # np.random.normal(scale=1.0, size=(15)) #*np.random.rand()
    r = r / np.dot(r, r)**0.5 * (3/16)**0.5*np.random.rand()# * p
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
    assert np.isclose(np.dot(state, state), 1), "state is not normed"
    return np.outer(np.conjugate(state), state) + 0.j

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
        #raise V    alueError("Pure State")
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
    ret = True
    sx = 0
    sy = 0
    x[::-1].sort()      # Sorting in descending order
    y[::-1].sort()
    for i in range(len(x)):
        sx += np.round(x[i], 4)
        sy += np.round(y[i], 4)
        ret = (ret and (sx <= sy))
    return ret

def check_majorisation_of_matrices(A: np.ndarray, B: np.ndarray) -> bool:
    """Check whether A ~ B, defined by \Lambda(A) ~ \Lambda(B), with \Lambda(X)
    the vector of all eigenvalues of X."""
    ew_A, ev = np.linalg.eig(A)
    ew_B, ev = np.linalg.eig(B)
    return check_majorisation_of_vectors(ew_A.real, ew_B.real)

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
    ew, ev = np.linalg.eig(rho)
    if (np.min(ew)) < 0:
        return False
    return True

def check_density_operator(rho):
    flag = check_density_operator_property_trace(rho)
    flag = (flag and check_density_operator_property_hermiticty(rho))
    flag = (flag and check_density_operator_property_positiv(rho))
    return flag

def kolmogorov_distance(rho1, rho2):
    ew, ev = np.linalg.eig(rho1 - rho2)
    return np.sum(np.abs(ew))/2.0

def trace_norm(A):
    return np.trace(matrix_root2(np.dot(np.transpose(A.conjugate()), A)))

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



def create_maximally_entangled_state(D: int, l: int, m: int) -> np.ndarray:
    """Creates a maximally entangled state according exercise 3.1 of Quantum
    Information 2018. D is the Dimension of H_A and H_B, such that the resulting
    state is in H_A \otimes H_B. For the case D = 2 it will be one of the Bell
    states. The selection which on is done by l,m \in 0,...,D-1"""
    assert l < D and l >= 0, "l have to be in 0,...,D-1"
    assert m < D and m >= 0, "m have to be in 0,...,D-1"
    base = np.eye(D)            # Create the computational basis
    phi = np.zeros(D**2) + 0.j  # initiate the resulting state
    for k in range(D):
        phi +=    np.exp(2*np.pi * 1.j * l * k / D) \
                * np.kron(base[k], base[(k - m) % D])   # state
    return phi/np.sqrt(D)                               # normation


def create_random_ensemble_arcsin(N: int, K: int, size: int) -> List[np.ndarray]:
    """TODO: Need more verification.
    @see: ZyPeNeCo2011"""
    rho_list = []
    n = int(N/2)
    phi1 = create_maximally_entangled_state(N, 0, 0)
    for i in range(size):
        U = np.zeros( (N**2, N**2) ) + 0.j
        for k in range(1, K):
            U += np.kron(np.eye(N), MF.make_matrix_random_unitary(N, N))
            #U += np.kron(MF.make_matrix_random_unitary(N, N), np.eye(N))
        phi2 = np.dot(U, phi1)

        psi = (phi1 + phi2)
        psi /= np.sqrt(np.dot(psi, psi))

        rho = density_matrix(psi)
        for k in range(n):
            rho = partial_trace(rho)/np.trace(rho)
        rho /= np.trace(rho)
        assert check_density_operator(rho), "rho is not a density matrix"
        rho_list.append(rho)

    return rho_list

def create_random_ensemble_ginibre(N:int, n = 4) -> List[np.ndarray]:
    """Creates an ensemble of density matrices with size N according to the
    Hilbert-Schmidt measure in the statespace of 2 qubit (4 x 4) using the
    construction by ginibre matrices."""
    return [make_random_density_matrix_from_ginibre(n) for i in range(N)]


def create_random_ensemble_pure(N:int, n = 4) -> List[np.ndarray]:
    list = []
    for i in range(N):
        rho = np.kron(  make_random_1qubit_density_matrix(1),
                        make_random_1qubit_density_matrix(1))
        U = MF.make_matrix_random_unitary(4, 4)
        rho = np.dot(U, np.dot(rho, np.conjugate(np.transpose(U))))
        list.append(rho)
    return list




def make_random_density_matrix_from_ginibre(N: int) -> np.ndarray:
    """Creates a random density matrix of size N from a ginibre matrix.
    @see MatrixFunctions.make_matrix_ginibre"""
    m = MF.make_matrix_ginibre(N)       # Defines Measure of matrix and positiv
    m2 = np.dot(np.conjugate(m.T),m)    # Make hermitian
    return m2 / np.trace(m2)            # Trace to one normation

def untangle_dist(rho: np.ndarray, dp: float) -> float:
    """Add a totally mixed density matrix to a entangled density matrix in a
    convex superposition and determine the point where the state become
    seperable."""
    p = 0
    con = concurrency(rho)
    while con > 0 and p < 1:
        con = concurrency((1 - p)*rho + p*np.eye(4)/4)
        p += dp

    return p

def qubit2_conv_comp_base_rep_to_hilbert_schmidt_rep(rho: np.ndarray):
    a = np.zeros(3) + 0.j
    b = np.zeros(3) + 0.j
    t = np.zeros( (3, 3) ) + 0.j
    for i in range(3):
        a[i] = np.trace(np.dot(rho, np.kron(Id, sigma[i])))
        b[i] = np.trace(np.dot(rho, np.kron(sigma[i], Id)))
        for j in range(3):
            t[i][j] = np.trace(np.dot(rho, np.kron(sigma[i], sigma[j])))

    return a, b, t

def qubit2_conv_comp_base_rep_to_svd_diag_rep(rho:np.ndarray):
    a, b, t = qubit2_conv_comp_base_rep_to_hilbert_schmidt_rep(rho)
    u, s, vh = np.linalg.svd(t)
    if MF.check_matrix_symmetric(t):
        print(concurrency(rho))
    if MF.check_matrixy_antisymmetric(t):
        print("nant")
    return np.dot(u.T, a).real, np.dot(vh, b).real, s, u, vh


def make_cue_matrix(N: int) -> np.ndarray:
    """
    mu = 0
    ep = 1/np.sqrt(N)
    sigma = ep**2 / 8
    K = np.random.normal(mu, sigma**0.5, (N ,N)) + 0.j
    H = K + np.transpose(K)"""
    m = MF.make_matrix_ginibre(N)
    H, R = np.linalg.qr(m)
    return H

def make_random_U_N(eps: float, N1, N2) -> np.ndarray:
    U = np.kron(make_cue_matrix(N1), make_cue_matrix(N2))
    U = np.dot(U, np.diag(np.exp(2.j*np.pi*eps*(np.random.random(N1 * N2)-0.5))))
    return U

def entropy_dist():
    lamb = np.linspace(0, 2, 50)
    av_entr = []
    for i, l in enumerate(lamb):
        U = make_random_U_N(np.sqrt(32*np.pi**4 * l / (4**4)))
        ew, ev = np.linalg.eig(U)
        entr = []
        for evv in ev:
            evv = evv / np.dot(evv, evv)**0.5
            rho = density_matrix(evv)
            entr.append(von_neuman_entropy(partial_trace(density_matrix(evv))))
        av_entr.append(np.mean(entr))

    plt.plot(lamb, av_entr)
    plt.show()






def commuter_matrix(m1, m2):
    return np.dot(m1, m2) - np.dot(m2, m1)
