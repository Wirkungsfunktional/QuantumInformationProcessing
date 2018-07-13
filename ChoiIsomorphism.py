import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def map(rho, alpha):
    return np.trace(rho)/2*np.eye(2) + alpha * ( \
        np.dot(QIF.sigma_x, np.dot(rho, QIF.sigma_z)) + \
        np.dot(QIF.sigma_z, np.dot(rho, QIF.sigma_x)))

def check_map_positiv():
    list = QIF.create_random_ensemble_ginibre(300, 2)
    N = 100
    alpha = np.linspace(0, 3.5, N)
    a_range = []
    for rho in list:
        flag = True
        i = 0
        while i < N and flag:
            new_rho = map(rho, alpha[i])
            ew, ev = np.linalg.eigh(new_rho)
            if (np.min(ew.real) >= 0 and np.trace(new_rho) <= 1.0):
                a_range.append(alpha[i])
            i += 1

    plt.hist(a_range)
    plt.show()

def plot_bloch_sphere():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    list = QIF.create_random_ensemble_ginibre(1000, 2)

    erg = []
    erg2 = []
    for i, rho in enumerate(list):
            new_rho = map(rho, 0.25)
            x = np.trace(np.dot(new_rho, QIF.sigma_x)).real
            y = np.trace(np.dot(new_rho, QIF.sigma_y)).real
            z = np.trace(np.dot(new_rho, QIF.sigma_z)).real
            erg.append([x, y, z])
            new_rho = map(rho, 0.1)
            x = np.trace(np.dot(new_rho, QIF.sigma_x)).real
            y = np.trace(np.dot(new_rho, QIF.sigma_y)).real
            z = np.trace(np.dot(new_rho, QIF.sigma_z)).real
            erg2.append([x, y, z])


    if len(erg) != 0:
        erg = np.array(erg)
        ax.scatter(erg[:,0], erg[:,1], erg[:,2])
    if len(erg2) != 0:
        erg2 = np.array(erg2)
        ax.scatter(erg2[:,0], erg2[:,1], erg2[:,2])

    r = np.sqrt(3)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    plt.show()

def choi_state(alpha):
    return np.eye(4)/4 + alpha/2 * (np.kron(QIF.sigma_x, QIF.sigma_z) + np.kron(QIF.sigma_z, QIF.sigma_x))

erg = []
for a in np.linspace(0, 1, 100):
    s = choi_state(a)
    ew, ev = np.linalg.eig(s)
    if np.min(ew) >= 0:
        erg.append(a)
plt.hist(erg)
plt.show()


plot_bloch_sphere()

check_map_positiv()
