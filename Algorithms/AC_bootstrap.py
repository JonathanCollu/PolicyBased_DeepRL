import torch
from copy import deepcopy
from Algorithms.PolicyBased import PolicyBased as PB

class ACBootstrap(PB):
    """ Parameters:
            - v_function: model in pytorch 
            - optimzer_v : value function optimization algorithm (torch optimizer)
            - n : estimation depth
            - maximize : if True V_phi(s) = max Q_theta(s, a) otherwise 
                         V_phi(s) = Q_theta(s,a) for a sampled by the current policy
    """
    def __init__(
            self,
            env, 
            model,
            optimizer,  
            epochs=10, 
            M=5, 
            T=500, 
            sigma=None,
            n=1, 
            maximize = True,
            baseline_sub = False):

        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.M = M
        self.T = T
        self.sigma = sigma
        self.n = n
        self.maximize = maximize
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
            for t in range(len(h0)):
                n = min(self.n, (len(h0) -1 -t)) # to avoid indexes out of bound
                v = self.val_fun.forward(h0[t + n][0], True)
                v = self.V(v, h0[t + n][1]) # take the max value or the value corresponding to the action sampled
                Q_n = sum([h0[t + k][3] for k in range(self.n)]) + v
                v_pred = self.val_fun.forward(h0[t][0], True)
                v_pred = self.V(v_pred, h0[t][1])
                
                if not self.baseline_sub: 
                    loss_policy += Q_n * h0[t][3]
                else:
                    loss_policy += (Q_n - v_pred.detach()) * h0[t][3]
                loss_value += torch.square(Q_n.detach() - v_pred)
        
        # compute the epoch's gradient and update policy head weights 
        self.train(self.model, loss_policy, self.optimizer)

        # compute the epoch's gradient and update the value head weights 
        self.train(self.val_fun, loss_value, self.optim_value)
        
        #return traces average loss and reward
        return loss_policy.item()/self.M, reward/self.M


    