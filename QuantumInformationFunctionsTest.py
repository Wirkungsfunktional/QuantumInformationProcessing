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

    def test_qbit_state_in_comp_basis(self):
        self.assertTrue(np.array_equal(QIF.q0, np.array([1, 0])))
        self.assertTrue(np.array_equal(QIF.q1, np.array([0, 1])))
        self.assertTrue(np.array_equal(QIF.q00, np.array([1, 0, 0, 0])))
        self.assertTrue(np.array_equal(QIF.q01, np.array([0, 1, 0, 0])))
        self.assertTrue(np.array_equal(QIF.q10, np.array([0, 0, 1, 0])))
        self.assertTrue(np.array_equal(QIF.q11, np.array([0, 0, 0, 1])))
        self.assertTrue(np.array_equal(QIF.phi_m, np.array([1,0,0,-1])/np.sqrt(2)))
        self.assertTrue(np.array_equal(QIF.phi_p, np.array([1,0,0, 1])/np.sqrt(2)))
        self.assertTrue(np.array_equal(QIF.psi_m, np.array([0,1,-1,0])/np.sqrt(2)))
        self.assertTrue(np.array_equal(QIF.psi_p, np.array([0,1, 1,0])/np.sqrt(2)))

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

    @unittest.expectedFailure
    def test_fidelity(self):
        rho = QIF.density_matrix(QIF.q00)
        totally_mixed = np.eye(4) / 4
        self.assertAlmostEqual(QIF.fidelity(rho, rho), 1.0)
        self.assertAlmostEqual(QIF.fidelity(rho, totally_mixed), 0.5)

    def test_check_purity(self):
        rho = QIF.density_matrix(QIF.q00)
        rho2 = 0.5*rho + 0.5*QIF.density_matrix(QIF.q11)
        totally_mixed = np.eye(4) / 4
        self.assertAlmostEqual( QIF.purity(rho), 1.0)
        self.assertNotEqual( QIF.purity(rho2) , 1.0)
        self.assertNotEqual( QIF.purity(totally_mixed) , 1.0)

    def test_qbit_from_bloch_sphere(self):
        theta = np.linspace(0, np.pi/2, 100)
        for t in theta:
            self.assertAlmostEqual(
                np.dot(     QIF.qbit_from_bloch_sphere(t, 0),
                            QIF.qbit_from_bloch_sphere(t, np.pi)),
                np.cos(t))

    def test_qbit_density_matrix(self):
        rho = QIF.qbit_density_matrix(np.array([1/np.sqrt(2), 1/2, 1/2]))
        self.assertTrue(QIF.check_density_operator_property_trace(rho))
        self.assertTrue(QIF.check_density_operator_property_hermiticty(rho))




    """
    Need PartialTrace in System B
    def test_locc_theorem(self):
        A = QIF.density_matrix(QIF.phi_p)
        theta = np.linspace(0, 2*np.pi, 100)
        for i, t in enumerate(theta):
            psi_end = np.cos(t) * QIF.q11 + np.sin(t) * QIF.q00
            B = QIF.density_matrix(psi_end)
            self.assertTrue( QIF.check_majorisation_of_matrices(A, B) )
    """

unittest.main()
