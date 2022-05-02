from ES_base_framework.Population import *
import numpy as np


class EA:
    """ Main Evolutionary Strategy class
    """
    def __init__(self, rl_alg, value_layer, minimize, budget,
                parents_size, offspring_size,
                individual_size,
                recombination, mutation, 
                selection, evaluation,
                verbose):
        self.rl_alg = rl_alg
        self.value_layer = value_layer
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

        layers_shape = []
        init_parent = []
        for name, params in self.rl_alg.model.state_dict().items():
            if self.rl_alg.use_es == 0 and self.value_layer and "value_layer" not in name:
                continue
            elif self.rl_alg.use_es == 1 and not self.value_layer and "policy_layer" not in name:
                continue
            layers_shape.append(list(params.shape))
            init_parent.append(params.cpu().numpy().flatten())
            
        self.parents = Population(  self.parents_size,
                                    self.individual_size,
                                    mutation,
                                    np.hstack(init_parent))
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
        self.parents.evaluate(self.evaluation, self.rl_alg, self.value_layer)
        best_eval, best_index = self.parents.best_fitness(self.minimize)
        best_indiv = self.parents.individuals[best_index]
        best_rew = self.parents.rewards[best_index]
        curr_budget += self.parents_size

        while curr_budget < self.budget:
            gen_tot += 1

            # Recombination: creates new offspring
            if self.recombination is not None and (self.parents_size > 1):
                self.recombination(self.parents, self.offspring)
            
            # Mutation: mutate individuals (offspring)
            self.mutation(self.offspring, gen_succ, gen_tot)

            # Evaluate offspring population
            self.offspring.evaluate(self.evaluation, self.rl_alg, self.value_layer)
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
                best_rew = self.parents.rewards[0]
                if self.verbose > 0:
                    loss_name = "value" if self.value_layer else "policy"
                    print(f"[{curr_budget}/{self.budget}] New best {loss_name} loss: {round(best_eval, 2)}" + \
                    f" | Mean reward: {best_rew} | P_succ: {round(gen_succ/gen_tot, 2)}")

        return best_indiv, best_eval, self.parents.alt_loss, best_rew