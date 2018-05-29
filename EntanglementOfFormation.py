import numpy as np
import scipy.linalg as sl
from matplotlib import pyplot as plt
import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF

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
    m = 0.5*QIF.density_matrix(QIF.phi_p) + 0.5*QIF.density_matrix(QIF.q00)
    print(np.linalg.eig(np.dot(m, np.dot( QIF.sigma_y_4d,np.dot(m , QIF.sigma_y_4d) ))))
    for i, pp in enumerate(p):
        w = pp*QIF.density_matrix(QIF.phi_p) + (1 - pp)*QIF.density_matrix(QIF.q00) #QIF.werner_state(pp, QIF.psi_m)
        #dist[i] = QIF.fidelity(w, np.dot(QIF.sigma_y_4d, np.dot(w, QIF.sigma_y_4d)))
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


def A15(x):
    return 1/(5*x)

def ensemble_test():
    e = []
    pur = []
    rho_check = np.zeros( (4, 4) ) + 0.j
    for rho in QIF.create_2qubit_random_density_matrix_ensemble(20000):
        rho_check += rho
        conc = QIF.concurrency(rho)
        pur.append(QIF.purity(rho).real)
        e.append(QIF.entanglement_2qbit(conc))
    e = np.array(e)
    notentangled = len(e) - np.count_nonzero(e)
    entangled = np.count_nonzero(e)
    n = len(e)
    print(notentangled/n)
    print(entangled/n)
    print(rho_check/n)
    plt.hist(pur, normed=True)#, bins=np.linspace(0.25, 1, 100))
    p = np.linspace(0.25, 1, 100)
    plt.plot(p, A15(p))
    plt.show()





def POVM_measurement():
    alpha = 1 * np.pi / 8
    u = np.array([np.cos(alpha), np.sin(alpha)])
    v = np.dot(QIF.sigma_x, u)
    rho = (QIF.density_matrix(u) + QIF.density_matrix(v))/2.0
    n = 1000
    x = np.linspace(0, 1, n)
    E_dk_expec = np.zeros(n)
    E_nu_expec = np.zeros(n)
    E_nv_expec = np.zeros(n)

    for i, xx in enumerate(x):
        E_dk = (1 - 2*xx)*np.eye(2) + xx*(QIF.density_matrix(u) + QIF.density_matrix(v))
        E_nu = xx*(np.eye(2) - QIF.density_matrix(u))
        E_nv = xx*(np.eye(2) - QIF.density_matrix(v))

        if np.linalg.det(E_dk) > 0:
            E_dk_expec[i] = np.trace(np.dot(rho, E_dk))
        if  QIF.check_minor_of_matrix(E_nu):
            E_nu_expec[i] = np.trace(np.dot(rho, E_nu))
        if QIF.check_minor_of_matrix(E_nv):
            E_nv_expec[i] = np.trace(np.dot(rho, E_nv))

    plt.plot(x, 1 - x*np.cos(2*alpha)**2, label="theorie")
    plt.plot(x, E_dk_expec, label="E_dk")
    plt.plot(x, E_nu_expec, label="E_nu")
    plt.plot(x, E_nv_expec, label="E_nv")
    plt.plot([1/(1 + np.sin(2*alpha))], [np.sin(2*alpha)], 'o')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    print(__doc__)
    #POVM_measurement()
    #plot_entanglement()
    ensemble_test()
    #copy_entanglement()
    #test_positivity()
