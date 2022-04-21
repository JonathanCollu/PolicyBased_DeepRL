import torch
from torch.distributions import Categorical

class REINFORCE:
    """ Parameters:
            - optimzer : optimization algorithm (torch optimizer)
            - epochs : number of epochs 
            - M : number of traces per epoch
            - n : trace length
            - gamma : discount factor
            - sigma : std in case of deterministic policy (if None the policy is stochastic)
            - model : differentiable parametrized policy (model in pytorch)
            - env : Environment to train our model 
    """
    def __init__(
            self, 
            env, 
            model,
            optimizer, 
            epochs=10, 
            M=5, 
            n=500,
            gamma=0.9, 
            sigma=None):

        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.M = M
        self.n = n
        self.gamma = gamma
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
        for i in range(self.n):
            a, lp = self.select_action(s)
            s_next, r, done, _ = self.env.step(a)
            trace.append((s, a, r, lp, s_next))
            s = s_next
            if done: s = self.env.reset()
        return trace


    def epoch(self):
        loss = 0 # initialize the epoch loss to 0
        for m in range(self.M):
            s = self.env.reset()
            h0 = self.sample_trace(s)
            R = 0
            for t in range(self.n - 1, -1, -1):
                R = h0[t][2] + self.gamma * R 
                loss += R * h0[t][3]
        
        # compute the epoch's gradient and update weights  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    