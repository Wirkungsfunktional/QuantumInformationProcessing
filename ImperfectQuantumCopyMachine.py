from qiskit import QuantumProgram
import numpy as np

__doc__=    """ The Program performs an imperfect copy of a qubit state
                $\ket{\Phi}$ using an additional single quantum register such
                that $\ket{\Phi 0 0}$ maps should map to $\ket{\Phi \Phi ?}$.
                Due to the no-cloning theorem this will not be possible.
                Nevertheless with some uncertanity it is possible to copy the
                state. For further information see:
                    @article{BuHi1996,
                      author = {V. Bužek and M. Hillery},
                      title = {Quantum copying: Beyond the no-cloning theorem},
                      journal = {Physical Review A},
                      year = {1996},
                      }
                and
                    @article{BuBrHiBr1997,
                      author = {V. Bužek and L. Braunstein and Hillery and D. Bruß},
                      title = {Quantum copying: A network},
                      journal = {Physical Review A},
                      year = {1997},
                    }
                """


qp = QuantumProgram()
qr = qp.create_quantum_register('qr',3)
cr = qp.create_classical_register('cr',3)
qc = qp.create_circuit('ImperfectQuantumCopy',[qr],[cr])

inp = input("val: ")


if inp == "000":
    pass
elif inp == "100":
    qc.x(qr[2])
elif inp == "H00":
    qc.h(qr[2])


a = 2
b = 1
c = 0

alpha = np.pi/8.0
beta = -np.sqrt(np.arcsin(0.5 - np.sqrt(2)/3.0))

qc.ry(alpha, qr[b])
qc.cx(qr[b], qr[c])
qc.ry(beta, qr[c])
qc.cx(qr[c], qr[b])
qc.ry(alpha, qr[b])

qc.cx(qr[a], qr[b])
qc.cx(qr[a], qr[c])
qc.cx(qr[b], qr[a])
qc.cx(qr[c], qr[a])

qc.measure(qr[a], cr[0])
qc.measure(qr[b], cr[1])
#qc.measure(qr[2], cr[2])


result = qp.execute('ImperfectQuantumCopy')
print(result.get_counts('ImperfectQuantumCopy'))
