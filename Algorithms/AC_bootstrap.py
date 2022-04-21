import torch
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
            v_func,
            optimizer_v,  
            epochs=10, 
            M=5, 
            T=500, 
            sigma=None,
            n=1, 
            maximize = True):

        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.M = M
        self.T = T
        self.sigma = sigma
        self.v_func = v_func
        self.optimizer_v = optimizer_v
        self.n = n
        self.maximize = maximize


    def V(self, v, h0, t):
        if self.maximize:
            return torch.max(v) # take the maximum Q_value
        return v[0][h0[t][1]] # take the Q_value of the action sampled

    def epoch(self):
        loss_policy = 0 # initialize the epoch loss_policy to 0
        loss_value = 0 # initialize the epoch loss_value to 0
        for _ in range(self.M):
            s = self.env.reset()
            h0 = self.sample_trace(s)
            for t in range(self.T):
                n = min(self.n, (self.T -1 -t))
                n += t
                v = self.V(self.v_func.forward(torch.tensor(h0[n][0]).unsqueeze(0)), h0, n)
                Q_n = sum([h0[t + k][3] for k in range(self.n)]) + v
                loss_policy += Q_n * h0[t][3]
                loss_value += torch.square(Q_n.detach() - self.V(self.v_func.forward(torch.tensor(h0[t][0]).unsqueeze(0)), h0, t))
        
        # compute the epoch's gradient and update policy weights 
        self.model.train() 
        self.optimizer.zero_grad()
        loss_policy.backward()
        self.optimizer.step()


        # compute the epoch's gradient and update the value function weights  
        self.v_func.train()
        self.optimizer_v.zero_grad()
        loss_value.backward()
        self.optimizer_v.step()

    