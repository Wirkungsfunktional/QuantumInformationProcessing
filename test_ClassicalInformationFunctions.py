import ClassicalInformationFunctions as CIF
import unittest
import numpy as np


class ClassicalInformationFunctionsTestCase(unittest.TestCase):

    def test_classical_entropy(self):
        # Test complete certainity
        e = CIF.classical_entropy(np.array([1, 0]))
        self.assertAlmostEqual(e, 0)

        # Test equal partition
        for i in range(2, 100):
            e = CIF.classical_entropy(np.array([1/i]*i))
            self.assertAlmostEqual(e, np.log2(i))

    def test_conditional_entropy(self):
        p_x = np.array([0.1, 0.3, 0.6])
        p_yx = np.array([   [0.70, 0.20, 0.10],
                            [0.05, 0.80, 0.15],
                            [0.10, 0.10, 0.80]])

        H = CIF.conditional_entropy(p_x, p_yx)
        self.assertAlmostEqual(H, 0.934, places=3)


    def test_mutual_information(self):
        p_x = np.array([0.1, 0.3, 0.6])
        p_yx = np.array([   [0.70, 0.20, 0.10],
                            [0.05, 0.80, 0.15],
                            [0.10, 0.10, 0.80]])
        H = CIF.mutual_information(p_x, p_yx)
        self.assertAlmostEqual(H, 0.479, places=3)

    def test_joint_entropy(self):
        m = np.array([  [1/8, 0, 1/8],
                        [1/16, 1/32, 1/32],
                        [0, 1/8, 0],
                        [1/8, 1/8, 1/4]])
        H = CIF.joint_entropy(m)
        self.assertAlmostEqual(H, 2.94, places=2)

    def test_get_conditional_prob_from_joint_prob(self):
        m = np.array([  [1/8, 0, 1/8],
                        [1/16, 1/32, 1/32],
                        [0, 1/8, 0],
                        [1/8, 1/8, 1/4]])
        m_ana = np.array([  [1/2, 0, 1/2],
                            [1/2, 1/4, 1/4],
                            [0, 1, 0],
                            [1/4, 1/4, 1/2]])
        m_res = CIF.get_conditional_prob_from_joint_prob(m)
        self.assertTrue(np.allclose(m_res, m_ana))

    def test_check_kraft_inequality_for_binary(self):
        code1 = {   "x1": "00",
                    "x2": "01",
                    "x3": "10",
                    "x4": "110",
                    "x5": "111",
                    "x6": "1101"}
        code2 = {   "x1": "00",
                    "x2": "01",
                    "x3": "10",
                    "x4": "110",
                    "x5": "1110",
                    "x6": "1111"}
        self.assertTrue(CIF.check_kraft_inequality(code2))
        self.assertFalse(CIF.check_kraft_inequality(code1))

    def test_get_alphabet(self):
        code1 = {   "x1": "00",
                    "x2": "01",
                    "x3": "10",
                    "x4": "110",
                    "x5": "111",
                    "x6": "1101"}
        code2 = {   "x1": "abc",
                    "x2": "01",
                    "x3": "ea"}
        alphabet1 = CIF.get_alphabet(code1)
        alphabet2 = CIF.get_alphabet(code2)
        self.assertTrue(len(alphabet1 ^ set(["0", "1"])) == 0)
        self.assertTrue(len(alphabet2 ^ set(["0", "1", "a", "b", "c", "e"])) == 0)


    def test_check_fano_condition(self):
        code1 = {   "x1": "00",
                    "x2": "01",
                    "x3": "10",
                    "x4": "110",
                    "x5": "111",
                    "x6": "1101"}
        code2 = {   "x1": "abc",
                    "x2": "01",
                    "x3": "ea"}

        self.assertTrue(CIF.check_fano_condition(code2))
        self.assertFalse(CIF.check_fano_condition(code1))







unittest.main()
