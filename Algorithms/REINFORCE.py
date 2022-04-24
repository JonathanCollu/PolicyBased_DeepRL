import torch
from copy import deepcopy
from Algorithms.PolicyBased import PolicyBased as PB
class REINFORCE(PB):

    def __init__(self, env, model, optimizer, epochs=10, M=5, T=500, gamma=0.9, sigma=None, baseline_sub=False, maximize=False):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.M = M
        self.T = T
        self.gamma = gamma
        self.sigma = sigma
        self.val_fun = None
        self.optim_value = None
        self.maximize = maximize
        if baseline_sub:
            self.val_fun = deepcopy(model)
            self.optim_value = torch.optim.Adam(self.val_fun.parameters(), lr = 0.001)

    def epoch(self):
        loss = 0 # initialize the epoch gradient to 0
        loss_value = 0 # initialize the epoch loss_value to 0
        reward = 0
        for _ in range(self.M):
            s = self.env.reset()
            h0, reward_t = self.sample_trace(s)
            reward += reward_t
            R = 0
            for t in range(len(h0) - 1, -1, -1):
                R = h0[t][2] + self.gamma * R 
                if self.val_fun:
                    v = self.val_fun.forward(h0[t][0], True)
                    v = self.V(v, h0[t][1])
                    R -= v.detach()
                    loss_value += torch.square(R.detach() - v)
                loss += R * h0[t][3]
        print(loss)
        # compute the epoch's gradient and update weights
        self.train(self.model, loss, self.optimizer)
        # if using baseline sub update value function model weights
        if loss_value: 
            self.train(self.val_fun, loss_value, self.optim_value) 
        #return traces average loss and reward
        return loss.item()/self.M, reward/self.M

    