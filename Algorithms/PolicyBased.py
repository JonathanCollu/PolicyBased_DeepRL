import torch
from torch.distributions import Categorical

class PolicyBased:
    """ Parameters:
            - optimzer : optimization algorithm (torch optimizer)
            - epochs : number of epochs 
            - M : number of traces per epoch
            - T : trace length
            - gamma : discount factor
            - sigma : std in case of deterministic policy (if None the policy is stochastic)
            - model : differentiable parametrized policy (model in pytorch)
            - env : Environment to train our model 
    """
    def __init__(self, env, model, epochs, M, T, sigma):
        self.env = env
        self.model = model
        self.epochs = epochs
        self.M = M
        self.T = T
        self.sigma = sigma

    def __call__(self):
        
        for epoch in range(self.epochs):
            self.epoch()
        return self.model # model or just its parameters?

    def select_action(self, s):

        # get the probability distribution of the actions
        dist = self.model.forward(torch.tensor(s).unsqueeze(0))
        # if the policy is deterministic add gaussian noise
        if self.sigma:
            dist += torch.normal(0, self.sigma)
        dist = Categorical(dist)
        action = dist.sample()

        #return action and -log(p(a))
        return action.item(), -dist.log_prob(action)


    def sample_trace(self, s):

        trace = []
        for i in range(self.T):
            a, lp = self.select_action(s)
            s_next, r, done, _ = self.env.step(a)
            trace.append((s, a, r, lp, s_next))
            s = s_next
            if done: s = self.env.reset()
        return trace


    def epoch(self):
        pass

    