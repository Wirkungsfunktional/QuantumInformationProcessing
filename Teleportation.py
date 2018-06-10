import QuantumInformationFunctions as QIF
import numpy as np
from matplotlib import pyplot as plt

x = 0.5
alpha = np.cos(x)
beta = np.sin(x)
p = 1.0
random_flag = 0
if random_flag:
    rho_a = QIF.make_random_1make_random_density_matrix(0.9)#
else:
    rho_a = np.array([[np.abs(alpha)**2, np.conjugate(beta)*alpha],[np.conjugate(alpha)*beta, np.abs(beta)**2]]) + 0.j
rho_aAB = np.kron(QIF.werner_state(p, QIF.psi_m), rho_a)
mess_op_a = np.kron(np.eye(2), QIF.density_matrix(QIF.psi_m))
mess_op_b = np.kron(np.eye(2), QIF.density_matrix(QIF.psi_p))
mess_op_c = np.kron(np.eye(2), QIF.density_matrix(QIF.phi_m))
mess_op_d = np.kron(np.eye(2), QIF.density_matrix(QIF.phi_p))


red_a = QIF.partial_trace(QIF.partial_trace((np.dot(rho_aAB, mess_op_a)) / np.trace(np.dot(rho_aAB, mess_op_a))))
red_b = QIF.partial_trace(QIF.partial_trace((np.dot(rho_aAB, mess_op_b)) / np.trace(np.dot(rho_aAB, mess_op_b))))
red_c = QIF.partial_trace(QIF.partial_trace((np.dot(rho_aAB, mess_op_c)) / np.trace(np.dot(rho_aAB, mess_op_c))))
red_d = QIF.partial_trace(QIF.partial_trace((np.dot(rho_aAB, mess_op_d)) / np.trace(np.dot(rho_aAB, mess_op_d))))

print(rho_a)
print(np.dot(QIF.sigma_z, np.dot(red_b, QIF.sigma_z)))
print(np.dot(QIF.sigma_x, np.dot(red_c, QIF.sigma_x)))
print(np.dot(QIF.sigma_y, np.dot(red_d, QIF.sigma_y)))

n = 100
p = np.linspace(0.001, 0.999, n)
err = np.zeros(n)
err_test = np.zeros(n)
for i, pp in enumerate(p):
    rho_aAB = np.kron(QIF.werner_state(pp, QIF.psi_m), rho_a)
    red_a = QIF.partial_trace(QIF.partial_trace((np.dot(rho_aAB, mess_op_a))))/ np.trace(np.dot(rho_aAB, mess_op_a))
    err[i] = QIF.fidelity_qubit(red_a, rho_a)
    #print((QIF.von_neuman_entropy(rho_aAB) - QIF.von_neuman_entropy(red_a)))
    err_test[i] = np.trace(QIF.matrix_root2((1 - pp)/2.0*rho_a + pp*np.dot(rho_a, rho_a)))**2


plt.plot(p, err, label="Fidelity")
if random_flag:
    plt.plot(p, err_test, label="Fidelity ana")
else:
    plt.plot(p, (1 + p)/2.0, label="Fidelity ana")
    plt.plot([1/3], [2/3], "o")
plt.xlabel(r"$p$")
plt.ylabel(r"$F$")
plt.legend()
plt.show()

ppp = 0.999
for i, pp in enumerate(p):
    rho_a = QIF.make_random_1qubit_density_matrix(pp**0.5)
    rho_aAB = np.kron(QIF.werner_state(ppp, QIF.psi_m), rho_a)
    red_a = QIF.partial_trace(QIF.partial_trace((np.dot(rho_aAB, mess_op_a))))/ np.trace(np.dot(rho_aAB, mess_op_a))
    fid = QIF.fidelity_qubit(red_a, rho_a)
    ew , ev = np.linalg.eig(rho_a)
    plt.plot([ew[0]], [fid], "ro")
plt.plot(p, (np.sqrt((1 - ppp)/2.0*p + ppp*(p**2)) + np.sqrt((1 - ppp)/2.0*(1 - p) + ppp*(1 - p)**2 ))**2, label="fidelity")
plt.xlabel("lambda")
plt.ylabel("Fidelity")
plt.show()
