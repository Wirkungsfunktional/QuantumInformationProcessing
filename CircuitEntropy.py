import QuantumInformationFunctions as QIF
import ClassicalInformationFunctions as CIF
import MatrixFunctions as MF
import numpy as np
from matplotlib import pyplot as plt



print("Entanglement of CNOT")
rho = QIF.density_matrix(QIF.q00)
CNOT = np.array( [  [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])
rho = np.dot(CNOT, np.dot(rho, np.conjugate(CNOT.T)))
print(QIF.entanglement_2qbit(QIF.concurrency(rho)))

print("Entanglement of cH")
rho = QIF.density_matrix(QIF.q00)
a = 1/np.sqrt(2)
cH = np.array( [    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, a, a],
                    [0, 0, a,-a]])
rho = np.dot(cH, np.dot(rho, np.conjugate(cH.T)))
print(QIF.entanglement_2qbit(QIF.concurrency(rho)))


H = np.array([[1,1],[1,-1]])/np.sqrt(2)
I =  np.eye(2)

print("Entanglement of H  CNOT")
rho = QIF.density_matrix(QIF.q11)
a = 1/np.sqrt(2)
H2 = np.kron(H, I)
U = np.dot(CNOT, H2)
print(U)
rho = np.dot(U, np.dot(rho, np.conjugate(U.T)))
print(QIF.entanglement_2qbit(QIF.concurrency(rho)))


print("Entanglement of H cH")
rho = QIF.density_matrix(QIF.q00)
U = np.dot(cH, H2)
rho = np.dot(U, np.dot(rho, np.conjugate(U.T)))
print(U)
print(QIF.entanglement_2qbit(QIF.concurrency(rho)))

"""
print("Entanglement of H cH by state representation")
p = np.dot(U, QIF.q00)
rho = QIF.density_matrix(p)
red_rho = QIF.partial_trace(rho)
print(QIF.von_neuman_entropy(red_rho))"""


print("Entanglement of HH  CNOT")
rho = QIF.density_matrix(QIF.q01)
a = 1/np.sqrt(2)
H2 = np.kron(H, H)
U = np.dot(CNOT, H2)
U = np.dot(np.kron(H, I), U)
rho = np.dot(U, np.dot(rho, np.conjugate(U.T)))
print(QIF.entanglement_2qbit(QIF.concurrency(rho)))
print(U)
