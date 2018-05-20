import numpy as np
from matplotlib import pyplot as plt
import ClassicalInformationFunctions as CIF

def fair_coin_toss():
    """Return the result of a fair coin toss: with 1/2: 0 and 1/2: 1"""
    return np.random.randint(2)

def number_of_toss_for_to_occure_bit():
    """Count the number of coin toss neccessary to get the first 1."""
    n = 1
    while True:
        if fair_coin_toss():
            return n
        n += 1

def ensemble_of_prob_exp(ensemble_size, func):
    """Perform the experiment: func() for ensemble_size times and return the
    list of each experiments outcome"""
    experiment_id = np.arange(ensemble_size)
    experiment_outcome = []
    for i in experiment_id:
        experiment_outcome.append(func())
    return np.array(experiment_outcome)

def probability_for_experiment(number_of_experiments, func):
    """Preform the func number_of_experiments times. count the number of each
    distinct outcome and return the list of unique outcome and its probability
    as separate arrays."""
    data, counts = np.unique(   ensemble_of_prob_exp(
                    number_of_experiments,
                    func), return_counts=True)
    return data, 1.0*counts/number_of_experiments
