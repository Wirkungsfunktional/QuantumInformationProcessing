import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF
import numpy as np
from matplotlib import pyplot as plt




def n_entr():
    N = 16
    U1 = QIF.make_state_standard_map(16, 10)
    U2 = QIF.make_state_standard_map(16, 9)
    erg_1 = []
    erg_2 = []
    erg_3 = []
    erg_4 = []
    erg_entr = []
    lamb = np.linspace(0, 2, 20)
    for e in lamb:

        U = QIF.make_random_U_N(np.sqrt(12*e/N**2), N, N)
        ew1, ev1 = np.linalg.eig(U)#np.kron(U1, U2))
        ev1 = ev1[np.where(ew1>-1)]
        ev1 = ev1[np.where(ew1<1)]
        erg1 = []
        erg2 = []
        erg3 = []
        erg4 = []
        for i in range(len(ev1)):
            state = ev1[:,i]
            state /= np.dot(np.conjugate(state), state)**0.5
            rho = np.outer(state, np.conjugate(state))
            red_rho = QIF.partial_trace3(rho, N)
            erg1.append(QIF.von_neuman_entropy(red_rho))
            rho_m = np.dot(red_rho, red_rho)
            erg2.append(((1 - np.trace(rho_m)).real)/(2 - 1))
            rho_m = np.dot(rho_m, red_rho)
            erg3.append(((1 - np.trace(rho_m)).real)/(3 - 1))
            rho_m = np.dot(rho_m, red_rho)
            erg4.append(((1 - np.trace(rho_m)).real)/(4 - 1))


        """
        erg3 = []
        for i in range(len(U)):
            H_state = np.ones(len(U))/np.sqrt(len(U))
            H_state[i] *= -1
            state = np.dot(U, H_state)
            state /= np.dot(np.conjugate(state), state)**0.5
            s, u, v = QIF.get_schmidt_decomposition(state, 4, 4)
            erg3.append(CIF.classical_entropy(s**2))
        entr2 = np.mean(erg3)
        erg_entr.append(entr2)
        """
        erg_1.append(np.mean(erg1)/np.log2(np.e))
        erg_2.append(np.mean(erg2))
        erg_3.append(np.mean(erg3))
        erg_4.append(np.mean(erg4))

    S1 = (np.log(N)-1/2)
    a1 = np.pi* np.pi**0.5
    S2 = (1 - 2*N**(1 - 2))/(2 - 1)
    a2 = np.pi* ( np.pi**0.5 / 2 / 1)
    S3 = (1 - 5*N**(1 - 3))/(3 - 1)
    a3 = np.pi* ( 3*np.pi**0.5 / 4/ 2)
    S4 = (1 - 14*N**(1 - 4))/(4 - 1)
    a4 = np.pi* ( 15 *np.pi**0.5 / 8 / 6)
    lamb_pr = np.linspace(0, 2, 1000)
    plt.plot(lamb_pr, 1 - np.exp(-a1/S1 * np.sqrt(lamb_pr)))
    plt.plot(lamb_pr, 1 - np.exp(-a2/S2 * np.sqrt(lamb_pr)))
    plt.plot(lamb_pr, 1 - np.exp(-a3/S3 * np.sqrt(lamb_pr)))
    plt.plot(lamb_pr, 1 - np.exp(-a4/S4 * np.sqrt(lamb_pr)))
    plt.plot(lamb, np.array(erg_1)/S1, "o")
    plt.plot(lamb, np.array(erg_2)/S2, "o")
    plt.plot(lamb, np.array(erg_3)/S3, "o")
    plt.plot(lamb, np.array(erg_4)/S4, "o")
    #plt.plot(lamb, np.array(erg_entr))
    plt.show()


    s, u, v = QIF.get_schmidt_decomposition(state, 4, 4)
    print(CIF.classical_entropy(s**2))

n_entr()
