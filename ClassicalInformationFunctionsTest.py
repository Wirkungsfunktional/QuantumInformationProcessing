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




unittest.main()
