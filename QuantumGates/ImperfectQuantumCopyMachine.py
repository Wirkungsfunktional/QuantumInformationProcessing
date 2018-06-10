import numpy as np
from functools import reduce
import QuantumAlgorithm as QA

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

print(__doc__)
alpha = np.pi/8.0
beta = -np.sqrt(np.arcsin(0.5 - np.sqrt(2)/3.0))

algo = QA.QuantumAlgorithm('ImperfectQuantumCopy', 3, 3)
#algo.invert_nth_qbit(0)

algo.set_nth_qbit(0, np.pi/2.0)

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
print("The circuit consists of " + str(algo.get_number_of_gates()) + " gates.")

print(algo.get_result())
