import ClassicalStochasticFunctions as CSF
import ClassicalInformationFunctions as CIF
import unittest
import numpy as np



class ClassicalStochasticFunctionsTestCase(unittest.TestCase):


    def test_coin_toss_distribution(self):
        """Test according to a problem taken from Thomas and ????. The classical
        entropy of the distribution of the first occurence of a head in the a
        fair coin toss will be computed by experiment. The analytic value should
        be 2."""
        data, prob = CSF.probability_for_experiment(
                            100000,
                            CSF.number_of_toss_for_to_occure_bit)
        self.assertAlmostEqual(CIF.classical_entropy(prob), 2.0, 1)


unittest.main()
