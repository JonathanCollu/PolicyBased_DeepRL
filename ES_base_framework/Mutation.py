import numpy as np
import random
import math
from ES_base_framework.Population import *


class Mutation:
    def mutate(self):
        """ Mutate the whole population
        """
        pass

    def __call__(self, *args):
        self.mutate(*args)


class IndividualSigma(Mutation):
    """ Individual sigma method.
    """
    def mutate(self, population: Population, *_):
        tau = 1 / np.sqrt(2 * np.sqrt(population.individuals.shape[1]))
        tau_prime = 1 / np.sqrt(2 * population.individuals.shape[1])
        # one draw from N(0, tau') per individual
        tau_prime_drawns = np.random.normal(0, tau_prime, size=population.sigmas.shape[0])
        tau_prime_drawns = tau_prime_drawns.reshape(-1, 1).repeat(population.sigmas.shape[1], axis=1)
        # one draw from N(0, tau) per sigma (individuals x components)
        tau_drawns = np.random.normal(0, tau, size=population.sigmas.shape)
        # mutate sigmas
        population.sigmas = population.sigmas * np.exp(tau_drawns + tau_prime_drawns)
        # mutate components
        variations = np.random.normal(0, population.sigmas)
        population.individuals += variations


# TODO add support for one-sigma
class OneFifth(Mutation):
    """ 1/5 success rule method.
    """
    def __init__(self, alt=False):
        if alt:
            self.mutate = self.mutate_alt

    def mutate(self, population: Population, gen_succ: int, gen_tot: int, *_):
        c = 0.95
        k = 40  # sigmas reset patience
        # reset sigmas
        if gen_tot % k == 0:
            population.init_sigmas()
        # increare sigmas (explore more)
        elif gen_succ/gen_tot > 0.20:
            population.sigmas /= c
        # decrease sigmas (exploit more)
        elif gen_succ/gen_tot < 0.20:
            population.sigmas *= c
        # mutate components
        variations = np.random.normal(0, population.sigmas)
        population.individuals += variations


class Correlated(Mutation):
    def mutate(self, population: Population, *_):
        lr = 1/np.sqrt(2*(np.sqrt(population.ind_dim)))
        lr_prime = 1/(np.sqrt(2*population.ind_dim))
        beta = math.pi/36
        normal_matr_prime = np.random.normal(0,lr_prime,1)

        for ind_idx in range(population.pop_size):
            for sigma in range(population.ind_dim):

                # Update our sigmas
                normal_matr = np.random.normal(0,lr,1)
                population.sigmas[ind_idx][sigma] = population.sigmas[ind_idx][sigma]*(
                            np.exp(normal_matr+normal_matr_prime))

                # Update angles
                alphas_noise = np.random.normal(0,beta,len(population.alphas[ind_idx]))
                population.alphas[ind_idx] = population.alphas[ind_idx] + alphas_noise

                # Check something, i dunno remember why tho
                population.alphas[ind_idx][population.alphas[ind_idx] > math.pi] = population.alphas[ind_idx][population.alphas[ind_idx] > math.pi] - 2*math.pi*np.sign(population.alphas[ind_idx][population.alphas[ind_idx] > math.pi])

                #Calculate C matrix
                count = 0
                C = np.identity(population.ind_dim)
                for i in range(population.ind_dim-1):
                    for j in range(i+1,population.ind_dim):
                        R = np.identity(population.ind_dim)
                        R[i,i] = math.cos(population.alphas[ind_idx][count])
                        R[j,j] = math.cos(population.alphas[ind_idx][count])
                        R[i,j] = -math.sin(population.alphas[ind_idx][count])
                        R[j,i] = math.sin(population.alphas[ind_idx][count])
                        C = np.dot(C, R)
                        count += 1
                s = np.identity(population.ind_dim)
                np.fill_diagonal(s, population.sigmas[ind_idx])
                C = np.dot(C, s)
                #print(f"max C: {np.max(C)}, min: {np.min(C)}")
                #C = np.abs(np.dot(C, C.T))

                # Update offspring
                sigma_std = np.random.multivariate_normal(mean=np.full((population.ind_dim),fill_value=0), cov=C)
                fix = np.array([ random.gauss(0,i) for i in sigma_std ])
                population.individuals[ind_idx] =  population.individuals[ind_idx] + fix
