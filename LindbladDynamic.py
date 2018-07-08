import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as si


def deriv(t, y):
    gamma = 0.5
    omega = 1
    H = omega/2 * np.array([[1, 0],[0, -1]]) + 0.j
    L = np.sqrt(gamma) * np.array([[0, 1],[0,  0]]) + 0.j
    Lh = np.conjugate(L.T)
    y = y.reshape( (2, 2) )
    A = np.dot(L, y)
    B = np.dot(y, Lh)
    rho = -1.j*(np.dot(H, y) - np.dot(y, H)) + \
        1/2 * ((np.dot(A, Lh) - np.dot(Lh, A)) + (np.dot(L, B) - np.dot(B, L)) )
    return rho.flatten()


def run():
    rho = QIF.density_matrix((QIF.q1 + QIF.q0) / np.sqrt(2)) + 0.j
    rho = QIF.density_matrix(QIF.q1) + 0.j
    y = si.solve_ivp(deriv, [0, 1000], rho.flatten())

    rx = []
    ry = []
    rz = []
    for i, rho in enumerate(y.y.T):
        rho = rho.reshape( (2, 2) )

        rx.append(np.trace(np.dot(rho, QIF.sigma_x)).real)
        ry.append(np.trace(np.dot(rho, QIF.sigma_y)).real)
        rz.append(np.trace(np.dot(rho, QIF.sigma_z)).real)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rx, ry, rz, color="r")

    r = 1
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    ax.plot_wireframe(x, y, z, color="k")
    plt.show()

run()
