import numpy as np


__doc__ = """Collection of functions used in classical information theory. The
implementation is not made in respect to efficiency. There are a lot of tools
already existence in scipy or numpy. This modul is only for learning purpous,
how to write such a code by myself. The learning outcome is a better
understanding of the Theory and not a fast and efficent library."""

def classical_entropy(p):
    # TODO: Better method for limit case lim x->0: x * log{1/x}
    p = p[np.nonzero(p)] # avoid the limit 0 * log{1 / 0}
    return np.dot(p, np.log2(1/p))
