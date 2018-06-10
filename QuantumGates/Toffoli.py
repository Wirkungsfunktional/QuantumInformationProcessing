from qiskit import QuantumProgram



qp = QuantumProgram()
qr = qp.create_quantum_register('qr',3)
cr = qp.create_classical_register('cr',3)
qc = qp.create_circuit('Toffoli_Gate',[qr],[cr])

inp = input("val: ")


if inp == "000":
    pass
elif inp == "001":
    qc.x(qr[0])
elif inp == "010":
    qc.x(qr[1])
elif inp == "011":
    qc.x(qr[0])
    qc.x(qr[1])
elif inp == "100":
    qc.x(qr[2])
elif inp == "101":
    qc.x(qr[2])
    qc.x(qr[0])
elif inp == "110":
    qc.x(qr[2])
    qc.x(qr[1])
elif inp == "111":
    qc.x(qr[2])
    qc.x(qr[0])
    qc.x(qr[1])

a = 2
b = 1
c = 0


qc.h(qr[c])
qc.cx(qr[b], qr[c])
qc.tdg(qr[c])
qc.cx(qr[a], qr[c])
qc.t(qr[c])
qc.cx(qr[b], qr[c])
qc.tdg(qr[c])
qc.cx(qr[a], qr[c])
qc.tdg(qr[b])
qc.t(qr[c])
qc.cx(qr[a], qr[b])
qc.tdg(qr[b])
qc.h(qr[c])
qc.cx(qr[a], qr[b])
qc.t(qr[a])
qc.s(qr[b])

qc.measure(qr[0], cr[0])
qc.measure(qr[1], cr[1])
qc.measure(qr[2], cr[2])


result = qp.execute('Toffoli_Gate')
print(result.get_counts('Toffoli_Gate'))
