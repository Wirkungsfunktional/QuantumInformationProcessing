import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as si
from scipy.optimize import curve_fit

__doc__ = """
Program to examine the entanglment development in the grover algorithm for
various N. The algorithm is implemented in matrix representation and the
entanglment is calculate by entropy of the reduced system. The system is reduced
towards a two qubit system. This kind of reduction does only make sense in case
for one search solution. If there is more than one possible solution, than the
problem becomes more complicated. Than we have to consider different partial
traces of all possible qubit combinations.
"""


def plot_ent():
    P = []
    ent = []
    ent2 = []
    entr = []
    entr2 = []
    print("4partit")
    N = 9
    state = QIF.make_n_dim_hadamard_state(N)
    state0 = np.copy(state)
    U = 2*QIF.density_matrix(state0) - np.eye(2**N) # U_i
    iter = int(np.pi/4 * np.sqrt(2**N)) * 2
    state_ind = -1
    for k in range(iter):
        state[state_ind] *= -1  # U_f
        rho = QIF.density_matrix(state)
        for i in range(N-2):
            rho = QIF.partial_trace(rho)
        print(rho)
        print("P = " + str(np.abs(state[state_ind])**2) )
        P.append(np.abs(state[state_ind])**2)
        ent.append(2**(N-1) * QIF.entanglement_2qbit(QIF.concurrency(rho)))
        entr.append(QIF.von_neuman_entropy(rho))
        state = np.dot(U, state)
        rho = QIF.density_matrix(state)
        for i in range(N-2):
            rho = QIF.partial_trace(rho)
        ent2.append(2**(N-1) * QIF.entanglement_2qbit(QIF.concurrency(rho)))
        entr2.append(QIF.von_neuman_entropy(rho))#QIF.entanglement_2qbit(QIF.concurrency(rho)))

    ent = np.array(ent)#*2**(0.9*N - 1)
    ent2 = np.array(ent2)#*2**(0.9*N - 1)
    plt.title("N = " + str(N), fontsize=30)
    plt.plot(range(iter), P, label="Prob")
    #plt.plot(range(iter), np.gradient(np.array(P)), label="deriv")
    #plt.plot(range(iter), ent - ent2, label="sub")
    plt.plot(range(iter), ent, label="E U_f")
    plt.plot(range(iter), ent2, label="E U_i")
    plt.plot(range(iter), entr, label="S U_f")
    plt.plot(range(iter), entr2, label="S U_i")
    plt.xlabel("Iterations", fontsize=30)
    plt.ylabel("Entanglement", fontsize=30)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=30)
    plt.show()


def plot_N():
    def f(x, a, b):
        return 2**(a*x + b)
    print("4partit")
    n = range(3, 11, 1)
    ent_max = []
    for N in n:
        ent = []
        state = QIF.make_n_dim_hadamard_state(N)
        state0 = np.copy(state)
        U = 2*QIF.density_matrix(state0) - np.eye(2**N)
        iter = int(np.pi/4 * np.sqrt(2**N)) + 1
        state_ind = 0
        for k in range(iter):
            state[state_ind] *= -1
            rho = QIF.density_matrix(state)
            for i in range(N-2):
                rho = QIF.partial_trace(rho)
            print("P = " + str(np.abs(state[state_ind])**2) )
            ent.append(QIF.von_neuman_entropy(rho))#QIF.entanglement_2qbit(QIF.concurrency(rho)))
            state = np.dot(U, state)

        ent_max.append(np.max(ent))

    popt, pcov = curve_fit(f, n, ent_max)
    print(popt)
    plt.plot(n, f(n, *popt) )
    plt.plot(n, ent_max)
    plt.show()




def plot_space():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    N = 7
    state = QIF.make_n_dim_hadamard_state(N)
    state0 = np.copy(state)
    U = 2*QIF.density_matrix(state0) - np.eye(2**N)
    iter = 2*int(np.pi/4 * np.sqrt(2**N))
    state_ind = 0
    erg = []
    for k in range(iter):
        state[state_ind] *= -1
        rho = QIF.density_matrix(state)
        for i in range(N-2):
            rho = QIF.partial_trace(rho)

        a, b, s, u, vh = QIF.qubit2_conv_comp_base_rep_to_svd_diag_rep(rho)
        erg.append(s)
        state = np.dot(U, state)

    if len(erg) != 0:
        erg = np.array(erg)
        ax.scatter(-erg[:,0], erg[:,1], erg[:,2])


    ax.plot([-1, 1],[-1, 1],[-1,-1], 'k')
    ax.plot([-1, 1],[-1,-1],[-1, 1], 'k')
    ax.plot([-1,-1],[-1, 1],[-1, 1], 'k')

    ax.plot([1, 1],[1,-1],[-1, 1], 'k')
    ax.plot([1,-1],[1, 1],[-1, 1], 'k')

    ax.plot([1,-1],[-1, 1],[1, 1], 'k')

    ax.plot([0, 0, 0, 0, 0],[0, 1, 0, -1, 0],[1, 0, -1, 0, 1], 'r')
    ax.plot([0, 1, 0, -1, 0],[0, 0, 0, 0, 0],[1, 0, -1, 0, 1], 'r')

    ax.plot([1,0,-1,0,1],[0,1,0,-1,0],[0,0,0,0,0], 'r')


    plt.show()

#plot_space()
plot_ent()
#plot_N()
