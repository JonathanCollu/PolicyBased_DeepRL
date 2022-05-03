import numpy as np
import torch


class Population:
    """ Attributes:
            - pop_size : size of population
            - input_ : path specifing the location of the input image
            - mutation : defines the mutation to be used in order to initialize parameters
    """
    def __init__(self, pop_size, ind_dim, mutation, init_parent=None):
        self.mutation = mutation
        self.pop_size = pop_size
        self.ind_dim = ind_dim
        self.init_parent = init_parent
        # initialize individual values
        if self.init_parent is not None:
            self.individuals = np.repeat(np.expand_dims(init_parent, axis=0), self.pop_size, axis=0)
        else:
            self.individuals = np.random.uniform(0, 1, size=(self.pop_size, self.ind_dim))
        # initialize fitnesses
        self.fitnesses = []
        # initialize policy loss
        self.l_p = []
        # initialize value loss
        self.l_v = []
        # initialize sigmas
        self.init_sigmas()
        # initialize alphas if necessary
        if self.mutation.__class__.__name__ == "Correlated":
            self.alphas = np.deg2rad(np.random.uniform(0,360, size=(self.pop_size, int((self.ind_dim*(self.ind_dim-1))/2))))
        
  
    def init_sigmas(self):
        """ Initialize sigma values depending on the mutation method of choice.
        """
        if self.mutation.__class__.__name__ == "OneSigma":
            self.sigmas = np.random.uniform(max(0, 
                                                np.min(self.individuals)/6), 
                                                np.max(self.individuals)/6, 
                                                size=self.pop_size)
        else:
            self.sigmas = np.random.uniform(max(0, 
                                                np.min(self.individuals)/6), 
                                                np.max(self.individuals)/6, 
                                                size=(self.pop_size, self.ind_dim))
        
    def max_fitness(self):
        """ Calculates the maximum fitness of the population.
            Return max fitness and its index
        """
        return np.max(self.fitnesses), np.argmax(self.fitnesses)

    def min_fitness(self):
        """ Calculates the minimum fitness of the population.
            Return min fitness and its index
        """
        return np.min(self.fitnesses), np.argmin(self.fitnesses)
        
    def best_fitness(self, minimize=True):
        """ Calculates the best fitness based on the problem.
            Return the best its fitness and index

            minimize: True if it is a minimization problem, False if maximization
        """
        if minimize:
            best_fitness, best_index = self.min_fitness()
        else:
            best_fitness, best_index = self.max_fitness()
        return best_fitness, best_index

    def evaluate(self, es):
        """ Evaluate the current population
        """
        self.fitnesses = []
        self.rewards = []
        for ind in self.individuals:
            es.load_weights_to_model(ind)
            l_p, l_v, r = es.evaluation(es.rl_alg)
            l_p, l_v = l_p.item(), l_v.item()
            self.fitnesses.append(-r)
            self.l_p.append(l_p)
            self.l_v.append(l_v)