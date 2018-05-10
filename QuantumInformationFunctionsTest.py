import QuantumInformationFunctions as QIF
import unittest
import numpy as np



class QuantumInformationFunctionsTestCase(unittest.TestCase):

    def test_Concurrency(self):
        w = QIF.werner_state(0.2, QIF.psi_m)
        conc = QIF.concurrency(w)
        self.assertTrue(conc == 0)

        w = QIF.werner_state(0.5, QIF.psi_m)
        conc = QIF.concurrency(w)
        self.assertTrue(conc != 0) #TODO: Calculate analytic value

    def test_Qbit_state_in_comp_basis(self):
        self.assertTrue(np.array_equal(QIF.q0, np.array([1, 0])))
        self.assertTrue(np.array_equal(QIF.q1, np.array([0, 1])))
        self.assertTrue(np.array_equal(QIF.q00, np.array([1, 0, 0, 0])))
        self.assertTrue(np.array_equal(QIF.q01, np.array([0, 1, 0, 0])))
        self.assertTrue(np.array_equal(QIF.q10, np.array([0, 0, 1, 0])))
        self.assertTrue(np.array_equal(QIF.q11, np.array([0, 0, 0, 1])))

    def test_majorisation(self):
        v1 = np.array([1, 2, 4])
        v2 = np.array([1, 2, 5])
        self.assertTrue(QIF.check_majorisation_of_vectors(v1, v2) == True)
        self.assertTrue(QIF.check_majorisation_of_vectors(v2, v1) == False)

        A = np.array([[1, 0],[0, 2]])
        B = np.array([[2, 0],[0, 3]])
        self.assertTrue(QIF.check_majorisation_of_matrices(A, B) == True)
        self.assertTrue(QIF.check_majorisation_of_matrices(B, A) == False)

    def test_shannon_1bit_entropy(self):
        self.assertAlmostEqual(QIF.shannon_1bit(0), 0)
        self.assertAlmostEqual(QIF.shannon_1bit(1), 0)
        self.assertAlmostEqual(QIF.shannon_1bit(0.5), 1)


unittest.main()
