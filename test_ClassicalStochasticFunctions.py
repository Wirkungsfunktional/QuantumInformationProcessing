import ClassicalStochasticFunctions as CSF
import ClassicalInformationFunctions as CIF
import unittest
import numpy as np



class ClassicalStochasticFunctionsTestCase(unittest.TestCase):


    def test_coin_toss_distribution(self):
        N = 100000
        data, prob = CSF.probability_for_experiment(
                            N,
                            CSF.number_of_toss_for_to_occure_bit)
        self.assertAlmostEqual(CIF.classical_entropy(prob), 2.0, 1)


unittest.main()
