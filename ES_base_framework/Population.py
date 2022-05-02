import numpy as np
import torch


class Population:
    """ Attributes:
            - pop_size : size of population
            - input_ : path specifing the location of the input image
            - mutation : defines the mutation to be used in order to initialize parameters
    """
    def __init__(self, pop_size, ind_dim, mutation):
        self.mutation = mutation
        self.pop_size = pop_size
        self.ind_dim = ind_dim

        # initialize individual values
        self.individuals = np.random.uniform(0, 1, size=(self.pop_size, self.ind_dim))
        # initialize fitnesses
        self.fitnesses = []
        # initialize rewards
        self.rewards = []
        # loss of value/policy net (the one not optimized)
        self.alt_loss = None
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

    def evaluate(self, evaluation, rl_alg, value_layer=False):
        """ Evaluate the current population
        """
        self.fitnesses = []
        self.rewards = []
        for ind in self.individuals:
            for name, params in rl_alg.model.state_dict().items():
                if value_layer:
                    layer = "value_layer"
                    out_dim = 1
                else:
                    layer = "policy_layer"
                    out_dim = 2
                if layer in name:
                    if params.numel() == out_dim:
                        weights = torch.tensor(ind[-out_dim:])
                    else:
                        weights = torch.tensor(ind[:-out_dim].reshape((out_dim, 64)))
                    rl_alg.model.load_state_dict({name: weights}, strict=False)
            l_p, l_v, r = evaluation(rl_alg)
            l_p, l_v = l_p.item(), l_v.item()
            if self.alt_loss is None: self.alt_loss = l_v if value_layer else l_p
            self.fitnesses.append(l_v if value_layer else l_p)
            self.rewards.append(r)