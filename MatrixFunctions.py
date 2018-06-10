import numpy as np
import scipy.linalg as sl


"""Check Tools -----------------------------------------------------------"""

def check_matrix_square(m):
    a, b = m.shape
    return (a == b)

def check_matrix_symmetric(m):
    assert check_matrix_square(m), "matrix is not a square matrix"
    return np.allclose(m, np.transpose(m))

def check_matrix_hermitian(m):
    assert check_matrix_square(m), "matrix is not a square matrix"
    return np.allclose(m, np.transpose(np.conjugate(m)))

def check_matrix_positiv(m):
    pass

def check_matrix_unitary(m):
    a, b = m.shape
    return (np.allclose(np.dot(m, np.conjugate(m.T)), np.eye(a)) and \
            np.allclose(np.dot(np.conjugate(m.T), m), np.eye(b)))

def check_matrix_normal(m):
    pass

def check_matrix_orthogonal(m):
    a, b = m.shape
    return (np.allclose(np.dot(m, m.T), np.eye(a)) and \
            np.allclose(np.dot(m.T, m), np.eye(b)))

def check_matrix_complex(m):
    pass

def check_matrix_diagonal(m):
    pass

def check_matrix_triagonal(m):
    z = np.zeros_like(m)
    return (np.allclose(np.triu(m, 1), z) or np.allclose(np.tril(m, -1), z))

def check_matrix_singular(m):
    pass

def check_matrix_simple_stochastic(m):
    """Check that the sum over each row is 1."""
    size_a, size_b = m.shape
    one_row = np.ones(size_b)
    res_vec = np.dot(m, one_row)
    return np.allclose(res_vec, np.ones(size_a))

def check_matrix_double_stochastic(m):
    """Check that the sum over each row is 1 and the sum over each column is 1."""
    return (check_matrix_simple_stochastic(m) and
            check_matrix_simple_stochastic(m.T))


"""Analyse Tools -----------------------------------------------------------"""

def analyse_matrix_sparse(m, rtol_places=8):
    a, b = m.shape
    N = a*b
    m = np.round(m, rtol_places)
    n = np.count_nonzero(m)
    return (n, N)



"""Creation Tools -----------------------------------------------------------"""


def make_matrix_random_unitary(n, m, mu_real=1, mu_imag=1):
    H = mu_real*np.random.randn(n, m) + mu_imag*1.j*np.random.randn(n, m)
    Q, R = sl.qr(H)
    return Q

def make_matrix_random_triagonal(n, m, mu_real=1, mu_imag=1):
    H = mu_real*np.random.randn(n, m) + mu_imag*1.j*np.random.randn(n, m)
    Q, R = sl.qr(H)
    return R
