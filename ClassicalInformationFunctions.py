import numpy as np
import scipy.stats as sst


__doc__ = """Collection of functions used in classical information theory. The
implementation is not made in respect to efficiency. There are a lot of tools
already existence in scipy or numpy. This modul is only for learning purpous,
how to write such a code by myself. The learning outcome is a better
understanding of the Theory and not a fast and efficent library."""

def classical_entropy(p):
    assert check_l1_norm(p), "p is not l1-normed"
    return sst.entropy(p, base=2)

def conditional_entropy(p_x, p_yx):
    assert check_simple_stochastic_matrix(p_yx), "conditional matrix p_yx is not \
    simple stochastic."
    assert check_l1_norm(p_x), "p_x is not l1-normed"
    H = 0
    for i, p in enumerate(p_x):
        H += p * classical_entropy(p_yx[i])
    return H

def check_simple_stochastic_matrix(m):
    """Check that the sum over each row is 1."""
    size_a, size_b = m.shape
    one_row = np.ones(size_b)
    res_vec = np.dot(m, one_row)
    return np.allclose(res_vec, np.ones(size_a))

def check_double_stochastic_matrix(m):
    """Check that the sum over each row is 1 and the sum over each column is 1."""
    size_a, size_b = m.shape
    one_row_a = np.ones(size_a)
    one_row_b = np.ones(size_b)
    res_vec_a = np.dot(np.transpose(m), one_row_a)
    res_vec_b = np.dot(m, one_row_b)
    return (np.allclose(res_vec_a, one_row_b) and
            np.allclose(res_vec_b, one_row_b))

def check_l1_norm(p):
    return np.isclose(np.sum(p), 1)

def check_joint_probability_matrix(m):
    return np.isclose(np.sum(m), 1.0)

def joint_entropy(p_xy):
    assert check_joint_probability_matrix(p_xy), "p_xy is not a joint probability \
    matrix."
    return classical_entropy(p_xy.flatten())

def get_conditional_prob_from_joint_prob(p_xy):
    assert check_joint_probability_matrix(p_xy), "p_xy is not a joint probability \
    matrix."
    size_a, size_b = p_xy.shape
    p_x = np.dot(p_xy, np.ones(size_b))
    assert check_l1_norm(p_x), "p_x is not l1-normed."
    for i in range(size_a):
        p_xy[i] *= 1/p_x[i]
    assert check_simple_stochastic_matrix(p_xy), "after calculation the matrix \
    is not a conditional probability matrix"
    return p_xy

def mutual_information(p_x, p_yx):
    assert check_simple_stochastic_matrix(p_yx), "conditional matrix p_yx is not \
    simple stochastic."
    assert check_l1_norm(p_x), "p_x is not l1-normed"
    p_y = np.dot(np.transpose(p_yx), p_x)
    assert check_l1_norm(p_y), "p_y is not l1-normed"
    return classical_entropy(p_y) - conditional_entropy(p_x, p_yx)

def information_flow(f, H):
    return f * H

def kullback_leibler_distance(p, q):
    assert check_l1_norm(p), "p is not l1-normed"
    assert check_l1_norm(q), "q is not l1-normed"
    return sst.entropy(p, q, base=2)

def get_alphabet(code_dict):
    alphabet = set()
    for word, code in code_dict.items():
        for letter in code:
            alphabet.add(letter)
    return alphabet

def check_kraft_inequality(code_dict):
    D = len(get_alphabet(code_dict))
    s = 0
    for word, code in code_dict.items():
        s += 1/D**len(code)
    return (s <= 1)
