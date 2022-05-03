from ES_base_framework.Population import *
import numpy as np


class EA:
    """ Main Evolutionary Strategy class
    """
    def __init__(self, rl_alg, mode, minimize, budget,
                parents_size, offspring_size,
                individual_size,
                recombination, mutation, 
                selection, evaluation,
                verbose):
        self.rl_alg = rl_alg
        self.mode = mode
        self.minimize = minimize
        self.budget = budget
        self.parents_size = parents_size
        self.offspring_size = offspring_size
        self.individual_size = individual_size
        self.recombination = recombination
        self.mutation = mutation
        self.selection = selection
        self.evaluation = evaluation
        self.verbose=verbose

        # load initial parent as the current weights of the policy/value net
        self.layers_shape = []
        init_parent = []
        if self.mode == "policy":
            self.model = self.rl_alg.model
        else:
            self.model = self.rl_alg.model_v
        for _, params in self.model.state_dict().items():
            self.layers_shape.append(list(params.shape))
            init_parent.append(params.cpu().numpy().flatten())
        init_parent = np.hstack(init_parent)
        
        self.parents = Population(self.parents_size,
                                  self.individual_size,
                                  mutation,
                                  init_parent)
        self.offspring = Population(self.offspring_size, 
                                    self.individual_size, 
                                    mutation)

    def run(self):
        """ Main function to run the Evolutionary Strategy
        """
        # Initialize budget and best evaluation (as worst possible)
        curr_budget = 0
        best_eval = np.inf if self.minimize else -np.inf

        # Initialize (generation-wise) success probability params
        # Success means finding a new best individual in a given gen. of offspring
        # gen_tot=num. of offspring gen., gen_succ=num. of successfull gen.
        gen_tot = 0
        gen_succ = 0

        # Initial parents evaluation step
        self.parents.evaluate(self)
        best_eval, best_index = self.parents.best_fitness(self.minimize)
        best_indiv = self.parents.individuals[best_index]
        best_l_p = self.parents.l_p[best_index]
        best_l_v = self.parents.l_v[best_index]
        curr_budget += self.parents_size

        while curr_budget < self.budget:
            gen_tot += 1

            # Recombination: creates new offspring
            if self.recombination is not None and (self.parents_size > 1):
                self.recombination(self.parents, self.offspring)
            
            # Mutation: mutate individuals (offspring)
            self.mutation(self.offspring, gen_succ, gen_tot)

            # Evaluate offspring population
            self.offspring.evaluate(self)
            curr_budget += self.offspring_size

            # Next generation parents selection
            self.selection(self.parents, self.offspring, self.minimize)

            # Update the best individual in case of success
            curr_best_eval = self.parents.fitnesses[0]
            success = False
            if self.minimize:
                if curr_best_eval < best_eval:
                    success = True
            else:
                if curr_best_eval > best_eval:
                    success = True
            if success:
                gen_succ += 1
                best_indiv = self.parents.individuals[0]
                best_eval = curr_best_eval
                best_l_p = self.parents.l_p[0]
                best_l_v = self.parents.l_v[0]
                if self.verbose > 0:
                    print(f"[{curr_budget}/{self.budget}] New best mean reward: {-best_eval}" + \
                    f" | Policy loss: {round(best_l_p, 2)} | Value loss: {round(best_l_v, 2)}" + \
                    f" | P_succ: {round(gen_succ/gen_tot, 2)}")

        # load best individual as weights of the model
        self.load_weights_to_model(best_indiv)
        return best_l_p, best_l_v, -best_eval

    def load_weights_to_model(self, weights):
        i = 0
        for name, _ in self.model.state_dict().items():
            ind_i = sum([np.prod(self.layers_shape[j]) for j in range(i)])
            inf_f = ind_i + np.prod(self.layers_shape[i])
            params = torch.tensor(weights[ind_i:inf_f].reshape(self.layers_shape[i]))
            self.model.load_state_dict({name: params}, strict=False)
            i += 1