import torch
from copy import deepcopy
from Algorithms.PolicyBased import PolicyBased as PB


class Reinforce(PB):
    def __init__(
            self, env, model, optimizer, epochs, M,
            gamma, baseline_sub, entropy_reg, entropy_factor):

        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.M = M
        self.T = None # get full episodes
        self.gamma = gamma
        if baseline_sub:
            self.val_fun = deepcopy(model)
            self.optim_value = torch.optim.Adam(self.val_fun.parameters(), lr = 0.001)
        else:
            self.val_fun = None
            self.optim_value = None
        self.entropy_reg = entropy_reg
        self.entropy_factor = entropy_factor

    def epoch(self):
        loss_policy = 0 # initialize the epoch gradient to 0
        loss_value = 0 # initialize the epoch loss_value to 0
        reward = 0
        for _ in range(self.M):
            s = self.env.reset()
            h0, reward_t = self.sample_trace(s)
            reward += reward_t
            R = 0
            # len-2 reason: -1 for having 0..len-1 and -1 for skipping last state
            for t in range(len(h0) - 2, -1, -1):
                R = h0[t][2] + self.gamma * R 
                if self.val_fun is not None:
                    v = self.val_fun.forward(h0[t][0], True)
                    loss_value += torch.square(R - v)
                    v = v.detach()
                else:
                    v = 0
                loss_policy += (R - v) * -h0[t][3].log_prob(h0[t][1])
                if self.entropy_reg:
                    loss_policy += self.entropy_factor * -torch.sum([p * torch.log(p) for p in h0[t][3].probs][0])
        loss_policy /= self.M
        loss_value /= self.M
        reward /= self.M
        # compute the epoch gradient and update weights
        self.train(self.model, loss_policy, self.optimizer)
        # if using baseline sub update value function model weights
        if self.val_fun is not None:
            self.train(self.val_fun, loss_value, self.optim_value) 
        #return traces average loss_policy and reward
        return loss_policy.item(), reward