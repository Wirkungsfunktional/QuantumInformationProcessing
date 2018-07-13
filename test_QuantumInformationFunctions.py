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

    def test_distance_function(self):
        rho1 = QIF.density_matrix(QIF.phi_m)
        rho2 = QIF.density_matrix(QIF.q11)

        self.assertAlmostEqual( QIF.kolmogorov_distance(rho1, rho2),
                                QIF.trace_norm(rho1 - rho2)/2.0)

    def test_partial_trace(self):
        rho = np.array([    [1, 2, 3, 4],
                            [2, 6, 7, 8],
                            [3, 7, 11, 12],
                            [4, 8, 12, 16]])
        rho_erg = np.array( [   [7, 11],
                                [11, 27]] )
        self.assertTrue(np.array_equal(QIF.partial_trace(rho), rho_erg))

    def test_partial_trace_random(self):
        m1 = QIF.make_random_2qubit_density_matrix(0.3)
        m2 = QIF.make_random_2qubit_density_matrix(0.3)
        m3 = np.kron(m1, m2)
        m4 = QIF.partial_trace(QIF.partial_trace(QIF.partial_trace(m3)))

        self.assertTrue(QIF.check_density_operator_property_trace(m4))
        self.assertTrue(QIF.check_density_operator_property_hermiticty(m4))
        self.assertTrue(QIF.check_density_operator_property_positiv(m4))

    def test_von_neuman_entropy(self):
        m = np.array([  [1 + 1/3, 1/np.sqrt(3) -1.j/np.sqrt(3)],
                        [1/np.sqrt(3) +1.j/np.sqrt(3), 1 - 1/3]])/2.0
        H = QIF.von_neuman_entropy(m)
        self.assertAlmostEqual(H, 0.3236, places=3)

        m = QIF.density_matrix(QIF.q11)
        H = QIF.von_neuman_entropy(m)
        self.assertAlmostEqual(H, 0, places=3)

    def test_create_maximally_entangled_state(self):
        self.assertTrue(np.allclose(QIF.create_maximally_entangled_state(2, 0, 0), QIF.phi_p))
        self.assertTrue(np.allclose(QIF.create_maximally_entangled_state(2, 1, 0), QIF.phi_m))
        self.assertTrue(np.allclose(QIF.create_maximally_entangled_state(2, 0, 1), QIF.psi_p))
        self.assertTrue(np.allclose(QIF.create_maximally_entangled_state(2, 1, 1), QIF.psi_m))

        phi = QIF.create_maximally_entangled_state(4, 0, 0)
        rho = QIF.density_matrix(phi)
        rho_a = QIF.partial_trace(rho)
        self.assertAlmostEqual(QIF.von_neuman_entropy(rho_a), 1.0)


    def test_check_density_matrix_half_classical(self):
        self.assertTrue(QIF.check_density_matrix_half_classical(QIF.make_special_state_half_classical()))

    def test_make_n_dim_hadamard_state(self):
        state0 = (QIF.q000 + QIF.q001 + QIF.q010 + QIF.q011 + QIF.q100 + QIF.q101 + QIF.q110 + QIF.q111)/np.sqrt(8)
        self.assertTrue(np.allclose(QIF.make_n_dim_hadamard_state(3), state0))


    def test_create_base_n_comp(self):
        check_base = [  np.array([1,0,0,0]),
                        np.array([0,1,0,0]),
                        np.array([0,0,1,0]),
                        np.array([0,0,0,1])]
        test_base = QIF.create_base_n_comp(2)
        for i, test_state in enumerate(test_base):
            self.assertTrue(np.allclose(test_state, check_base[i]))














unittest.main()
