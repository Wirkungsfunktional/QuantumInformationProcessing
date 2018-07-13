from qiskit import QuantumProgram
import numpy as np

"""Program to perform the Deutsch Algorithm on an unknown (randomly selected)
one variable function f:{0, 1} -> {0, 1}. The Algorithm determine whether the
function is constant or balanced. """


qp = QuantumProgram()
qr = qp.create_quantum_register('qr',3)
cr = qp.create_classical_register('cr',3)
qc = qp.create_circuit('Deutsch',[qr],[cr])



def U(qc, qr, opt):
    if opt == 1:
        qc.x(qr[0])
        qc.x(qr[1])
        qc.ccx(qr[0], qr[1], qr[2])
        qc.x(qr[0])
        qc.x(qr[1])
    if opt == 2:
        qc.x(qr[0])
        qc.ccx(qr[0], qr[1], qr[2])
        qc.x(qr[0])
    if opt == 3:
        qc.x(qr[1])
        qc.ccx(qr[0], qr[1], qr[2])
        qc.x(qr[1])
    if opt == 4:
        qc.ccx(qr[0], qr[1], qr[2])


# Set the initial state |10>
qc.x(qr[2])

# Random choosen function
flag = np.random.randint(1, 5)




# Implement the Algorithm
qc.h(qr[0])
qc.h(qr[1])
qc.h(qr[2])

U(qc, qr, flag)
qc.h(qr[2])
"""
qc.h(qr[0])
qc.h(qr[1])

qc.x(qr[0])
qc.x(qr[1])

qc.h(qr[1])
qc.cx(qr[0], qr[1])
qc.h(qr[1])

qc.x(qr[0])
qc.x(qr[1])

qc.h(qr[0])
qc.h(qr[1])
"""
qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.h(qr[0])


qc.measure(qr, cr)
result = qp.execute('Deutsch')


# Output
print("Random selected Function is: ")
print("f" + str(flag))
print("Algorithm gives: ")
res = result.get_counts('Deutsch')
print(res)
