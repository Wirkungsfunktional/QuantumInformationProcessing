import MatrixFunctions as MF
import unittest
import numpy as np


class MatrixFunctionsTestCase(unittest.TestCase):

    def test_check_matrix_unitary(self):
        m1 = np.array([[1 + 1.j, 2, 3], [4, 5, 6]])
        m2 = MF.make_matrix_random_unitary(5, 6)

        self.assertFalse(MF.check_matrix_unitary(m1))
        self.assertTrue(MF.check_matrix_unitary(m2))

    def test_check_matrix_triagonal(self):
        m1 = MF.make_matrix_random_triagonal(3, 5)
        m2 = MF.make_matrix_random_triagonal(3, 5).T
        self.assertTrue(MF.check_matrix_triagonal(m1))
        self.assertTrue(MF.check_matrix_triagonal(m2))
        m3 = MF.make_matrix_random_unitary(5, 5)
        self.assertFalse(MF.check_matrix_triagonal(m3))

    def test_analyse_matrix_sparse(self):
        m1 = np.zeros( (100, 100) )
        m1[0][0] = 0.0004
        non_zero, N_total = MF.analyse_matrix_sparse(m1)
        self.assertTrue( non_zero == 1)
        self.assertTrue( N_total == 100*100)
        non_zero, N_total = MF.analyse_matrix_sparse(m1, rtol_places=2)
        self.assertTrue( non_zero == 0)
        m2 = np.ones( (20, 20) )
        non_zero, N_total = MF.analyse_matrix_sparse(m2, rtol_places=2)
        self.assertTrue( non_zero == N_total)

    def test_check_matrix_simple_stochastic(self):
        m =    np.array([   [0.70, 0.20, 0.10],
                            [0.05, 0.80, 0.15],
                            [0.10, 0.10, 0.80]])
        self.assertTrue(MF.check_matrix_simple_stochastic(m))
        m = np.array([[1.0, 0.2, 0.3],[0.0, 0.8, 0.7]])
        self.assertFalse(MF.check_matrix_simple_stochastic(m))


    def test_check_matrix_double_stochastic(self):
        m =    np.array([   [0.70, 0.20, 0.10],
                            [0.05, 0.80, 0.15],
                            [0.10, 0.10, 0.80]])
        self.assertFalse(MF.check_matrix_double_stochastic(m))
        m = np.array([  [0.5, 0.5],
                        [0.5, 0.5]])
        self.assertTrue(MF.check_matrix_double_stochastic(m))

    def test_check_matrixy_antisymmetric(self):
        m = np.array([  [0, 1, 2, 3],
                        [-1, 0,-4,-5],
                        [-2, 4, 0,-6],
                        [-3, 5, 6, 0]])
        self.assertTrue(MF.check_matrixy_antisymmetric(m))
        self.assertFalse(MF.check_matrixy_antisymmetric(np.eye(2)))

    def test_check_matrix_diagonal(self):
        m = np.eye(5)
        self.assertTrue(MF.check_matrix_diagonal(m))
        m[0][1] = 1
        self.assertFalse(MF.check_matrix_diagonal(m))




unittest.main()
