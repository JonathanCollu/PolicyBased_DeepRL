import numpy as np
from ES_base_framework.Population import *


class Selection:
    def __call__(self):
        pass


# TODO add maximization case (argsort)
class PlusSelection(Selection):
    """ Get the best individuals from both the parent and offspring populations
    """
    def __call__(self, parents: Population, offspring: Population, minimize=True):
        fitnesses_stacked = np.hstack([parents.fitnesses, offspring.fitnesses])
        l_p_stacked = np.hstack([parents.l_p, offspring.l_p])
        l_v_stacked = np.hstack([parents.l_v, offspring.l_v])
        # get sorted indexes
        if minimize:
            sorted_ind = np.argsort(fitnesses_stacked)[:parents.pop_size]
        else: 
            sorted_ind = np.argsort(fitnesses_stacked)[::-1][:parents.pop_size]
        # update parent population
        parents.individuals = np.vstack([parents.individuals, offspring.individuals])[sorted_ind]
        parents.sigmas = np.vstack([parents.sigmas, offspring.sigmas])[sorted_ind]
        parents.fitnesses = fitnesses_stacked[sorted_ind]
        parents.l_p = l_p_stacked[sorted_ind]
        parents.l_v = l_v_stacked[sorted_ind]
        if parents.mutation.__class__.__name__ == "Correlated":
                parents.alphas = np.vstack([parents.alphas, offspring.alphas])[sorted_ind]

# TODO add maximization case (argsort)
class CommaSelection(Selection):
    """ Get the best individuals from the offspring population
    """
    def __call__(self, parents: Population, offspring: Population, minimize=True):
        if minimize:
            sorted_ind = np.argsort(offspring.fitnesses)[:parents.pop_size]
        else:
            sorted_ind = np.argsort(offspring.fitnesses)[::-1][:parents.pop_size]
        # update parent population
        parents.individuals = offspring.individuals[sorted_ind]
        parents.sigmas = offspring.sigmas[sorted_ind]
        parents.fitnesses = offspring.fitnesses[sorted_ind]
        parents.l_p = offspring.l_p[sorted_ind]
        parents.l_v = offspring.l_v[sorted_ind]
        if parents.mutation.__class__.__name__ == "Correlated":
                parents.alphas = offspring.alphas[sorted_ind]