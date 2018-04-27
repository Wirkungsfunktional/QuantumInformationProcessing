try:
    import qiskit
    from qiskit import QuantumProgram
except ImportError:
    print("Module qiskit does not exists")
    exit()
if (qiskit.__version__ != "0.4.8"):
    print("Module version of qiskit is not tested. There could occure Problems.")

import numpy as np
from functools import reduce

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


class QuantumAlgorithm:
    def __init__(self, name, N_qm, N_cl):
        self.name = name
        self.N_qm = N_qm
        self.N_cl = N_cl
        self.input_state = [0]*self.N_qm
        self.quantum_program = QuantumProgram()
        self.quantum_register = self.quantum_program.create_quantum_register('qr',3)
        self.classical_register = self.quantum_program.create_classical_register('cr',3)
        self.quantum_circuit = self.quantum_program.create_circuit(
                                                    self.name,
                                                    [self.quantum_register],
                                                    [self.classical_register])
        self.result = None

    def invert_nth_qbit(self, n):
        """Invert the qbit at position n. Invert means 0 to 1 and 1 to 0. This
        is done by using the pauli_x matrix."""
        assert n >= 0 , "negativ index"
        assert n < self.N_qm, "index above size of register"
        self.quantum_circuit.x(self.quantum_register[n])
        self.input_state[n] = self.input_state[n] ^ 1 #Inversion by XOR

    def get_input_state(self):
        return self.input_state


    def measure_nth_qbit(self, n):
        assert n >= 0 , "negativ index"
        assert n < self.N_qm, "index above size of quantum register"
        assert n < self.N_cl, "index above size of classical register"
        self.quantum_circuit.measure(   self.quantum_register[n],
                                        self.classical_register[n])

    def run(self):
        self.result = self.quantum_program.execute(self.name)

    def get_result(self):
        return self.result.get_counts(self.name)

    def add_gate(self, name, register_list, *arg):
        if name == "pauli_x":
            self.quantum_circuit.x(self.quantum_register[register_list[0]])
        elif name == "pauli_y":
            self.quantum_circuit.y(self.quantum_register[register_list[0]])
        elif name == "pauli_z":
            self.quantum_circuit.z(self.quantum_register[register_list[0]])
        elif name == "hadamard":
            self.quantum_circuit.h( self.quantum_register[register_list[0]])
        elif name == "CNOT":
            self.quantum_circuit.cx(    self.quantum_register[register_list[0]],
                                        self.quantum_register[register_list[1]])
        elif name == "rotate_y":
            self.quantum_circuit.ry(    arg[0],
                                        self.quantum_register[register_list[0]])


print(__doc__)
alpha = np.pi/8.0
beta = -np.sqrt(np.arcsin(0.5 - np.sqrt(2)/3.0))

algo = QuantumAlgorithm('ImperfectQuantumCopy', 3, 3)
#algo.invert_nth_qbit(0)

algo.add_gate("rotate_y", [1], alpha)
algo.add_gate("CNOT", [1, 2])
algo.add_gate("rotate_y", [1], beta)
algo.add_gate("CNOT", [2, 1])
algo.add_gate("rotate_y", [1], alpha)

algo.add_gate("CNOT", [0, 1])
algo.add_gate("CNOT", [0, 2])
algo.add_gate("CNOT", [1, 0])
algo.add_gate("CNOT", [2, 0])

algo.measure_nth_qbit(0)
algo.measure_nth_qbit(1)
algo.measure_nth_qbit(2)

algo.run()

input_state = algo.get_input_state()

print("Initial state is: " + reduce(lambda x, y: str(x) + str(y),  input_state))


print(algo.get_result())
