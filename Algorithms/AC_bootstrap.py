import torch
from copy import deepcopy
from Algorithms.PolicyBased import PolicyBased as PB


class ACBootstrap(PB):
    """ Parameters:
            - v_function: model in pytorch 
            - optimzer_v : value function optimization algorithm (torch optimizer)
            - n : estimation depth
    """
    def __init__(
            self, env, model, optimizer, epochs,
            M, T, n, sigma, baseline_sub):

        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.M = M
        self.T = T
        self.n = n
        self.sigma = sigma
        self.baseline_sub = baseline_sub
        self.val_fun = deepcopy(model)
        self.optim_value = torch.optim.Adam(self.val_fun.parameters(), lr = 0.001) 

    def epoch(self):
        loss_policy = 0 # initialize the epoch loss_policy to 0
        loss_value = 0 # initialize the epoch loss_value to 0
        reward = 0
        for _ in range(self.M):
            s = self.env.reset()
            h0, reward_t = self.sample_trace(s)
            reward += reward_t
            for t in range(len(h0) - 1): # -1 for skipping last state
                n = min(self.n, (len(h0) - 1 - t)) # to avoid indexes out of bound
                v = self.val_fun.forward(h0[t + n][0], True)
                Q_n = sum([h0[t + k][2] for k in range(n - 1)]) + v
                v_pred = self.val_fun.forward(h0[t][0], True)
                if not self.baseline_sub:
                    loss_policy += Q_n.detach() * h0[t][3]
                else:
                    loss_policy += (Q_n.detach() - v_pred.detach()) * h0[t][3]
                loss_value += torch.square(Q_n.detach() - v_pred)
        loss_policy /= self.M
        loss_value /= self.M
        reward /= self.M

        # compute the epoch gradient and update policy head weights 
        self.train(self.model, loss_policy, self.optimizer)

        # compute the epoch gradient and update the value head weights 
        self.train(self.val_fun, loss_value, self.optim_value)
        
        #return traces average loss and reward
        return loss_policy.item(), reward