import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def test_geometry():
    N = 1000
    """
    list = QIF.create_random_ensemble_ginibre(N)
    #list = QIF.create_2qubit_random_density_matrix_ensemble(1, N)
    #list = QIF.create_2qubit_random_density_matrix_ensemble_by_random_unitary(N, 1)
    #list = QIF.create_random_ensemble_arcsin(4, 2, N)
    erg1 = []
    erg2 = []
    for i, rho in enumerate(list):
        a, b, s, u, vh = QIF.qubit2_conv_comp_base_rep_to_svd_diag_rep(rho)
        np.random.shuffle(s)
        if (QIF.concurrency(rho) == 0):
            rho2 = rho
            erg1.append(s)
        else:
            erg2.append(s)

    """
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(223, projection='3d')
    ax3 = fig.add_subplot(224, projection='3d')


    #rho2 = QIF.create_random_ensemble_ginibre(1)[0]
    rho2 = QIF.make_pure_random_2qbit_density_matrix_by_unitary_partial_trace(1, 1)
    rho3 = QIF.partial_transpose(np.copy(rho2))
    #rho2 = QIF.density_matrix((QIF.q00 + QIF.q10 + QIF.q01)/np.sqrt(3))
    """
    erg1 = []
    erg2 = []
    y = si.solve_ivp(deriv, [0, 1000], rho2.flatten())
    for i, rho in enumerate(y.y.T):
        rho = rho.reshape( (4, 4) )
        a, b, s, u, vh = QIF.qubit2_conv_comp_base_rep_to_svd_diag_rep(rho)
        if (QIF.concurrency(rho) == 0):
            erg1.append(s)
        else:
            erg2.append(s)

    erg1 = np.array(erg1)
    erg2 = np.array(erg2)
    if len(erg1) != 0:
        ax.scatter(-erg1[:,0], erg1[:,1], erg1[:,2])
    if len(erg2) != 0:
        ax.scatter(-erg2[:,0], erg2[:,1], erg2[:,2])

    """
    erg1 = []
    erg2 = []
    ergpt = []
    erg3 = []
    erg4 = []
    erg5 = []
    erg6 = []
    a = 6
    rr = (6*QIF.density_matrix(QIF.phi_m)/12 \
        + 2*QIF.density_matrix(QIF.q01)/12  + 2*QIF.density_matrix(QIF.q11)/12 + 2*QIF.density_matrix(QIF.q10)/12 )
    for i, p in enumerate(np.linspace(0, 1, 1000)):
        rho = p*rr + (1 - p)*np.eye(4)/4
        #rho = p*QIF.density_matrix((QIF.q00 + (QIF.q10 + QIF.q01)/np.sqrt(2))/np.sqrt(2)) + (1 - p)*rho2#QIF.density_matrix((QIF.psi_p))#rho2
        a, b, s, u, vh = QIF.qubit2_conv_comp_base_rep_to_svd_diag_rep(rho)
        #np.random.shuffle(a)
        #np.random.shuffle(b)
        #np.random.shuffle(s)
        if (QIF.concurrency(rho) == 0):
            erg1.append(s)
            erg3.append(a)
            erg4.append(b)
        else:
            #erg1.append(s)
            erg2.append(s)
            erg5.append(a)
            erg6.append(b)

    ergpt = np.array(ergpt)
    #ax.scatter(-ergpt[:,0], ergpt[:,1], ergpt[:,2])
    if len(erg1) != 0:
        erg1 = np.array(erg1)
        ax.scatter(-erg1[:,0], erg1[:,1], erg1[:,2])
    if len(erg2) != 0:
        erg2 = np.array(erg2)
        ax.scatter(-erg2[:,0], erg2[:,1], erg2[:,2])
    if len(erg3) != 0:
        erg3 = np.array(erg3)
        ax2.scatter(-erg3[:,0], erg3[:,1], erg3[:,2], "r")
    if len(erg4) != 0:
        erg4 = np.array(erg4)
        ax3.scatter(-erg4[:,0], erg4[:,1], erg4[:,2], "r")
    if len(erg5) != 0:
        erg5 = np.array(erg5)
        ax2.scatter(-erg5[:,0], erg5[:,1], erg5[:,2], "b")
    if len(erg6) != 0:
        erg6 = np.array(erg6)
        ax3.scatter(-erg6[:,0], erg6[:,1], erg6[:,2], "b")

    r = np.sqrt(3)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    ax2.plot_wireframe(x, y, z, color="r")
    ax3.plot_wireframe(x, y, z, color="r")
    #ax.plot_wireframe(x - 1, y + 1, z + 1, color="r")


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


test_geometry()
