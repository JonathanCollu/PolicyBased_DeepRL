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
    def __init__(self, env, model, epochs, M, T, sigma, maximize):
        self.env = env
        self.model = model
        self.epochs = epochs
        self.M = M
        self.T = T
        self.sigma = sigma
        self.maximize = maximize

    def __call__(self):
        rewards = []
        losses = []
        for epoch in range(self.epochs):
            l, r = self.epoch()
            print(f"[{epoch+1}] Epoch mean loss: {round(l, 4)} | Epoch mean reward: {r}")
            losses.append(l)
            rewards.append(r)
        return rewards

    def select_action(self, s):

        # TODO: implement entropy regularization

        # get the probability distribution of the actions
        dist = self.model.forward(s)

        # if the policy is deterministic add gaussian noise
        if self.sigma is not None:
            dist += torch.normal(0, self.sigma)

        # sample action from distribution
        dist = Categorical(dist)
        action = dist.sample()

        #return action and -log(p(a))
        return action.item(), -dist.log_prob(action)

    def sample_trace(self, s):
        reward = 0
        trace = []
        i = 0
        while True:
            if self.T is not None and i >= self.T:
                break
            i += 1
            a, lp = self.select_action(s)
            s_next, r, done, _ = self.env.step(a)
            trace.append((s, a, r, lp))
            reward += r
            s = s_next
            if done: break
        trace.append((s, None, None, None))
        return trace, reward

    def train(self, model, loss, opt):
        # set model to train
        model.train()
        # compute gradient of loss
        opt.zero_grad()
        loss.backward()
        # update weigths
        opt.step()

    def epoch(self):
        pass