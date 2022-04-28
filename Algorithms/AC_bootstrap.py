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
            M, T, n, baseline_sub, entropy_reg, entropy_factor, 
            val_fun, optimizer_v, run_name=None, device=None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Computing on {self.device} device")
        self.env = env
        self.model = model.to(device)
        self.optimizer = optimizer
        self.epochs = epochs
        self.M = M
        self.T = T
        self.n = n
        self.baseline_sub = baseline_sub
        self.entropy_reg = entropy_reg
        self.entropy_factor = entropy_factor
        self.val_fun = val_fun
        self.optim_value = optimizer_v
        self.run_name = run_name 

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
                v = self.val_fun.forward(h0[t + n][0], self.device, True)
                Q_n = sum([h0[t + k][2] for k in range(n - 1)]) + v
                v_pred = self.val_fun.forward(h0[t][0], self.device, True)
                if not self.baseline_sub:
                    loss_policy += Q_n.detach() * -h0[t][3].log_prob(h0[t][1])
                else:
                    loss_policy += (Q_n.detach() - v_pred.detach()) * -h0[t][3].log_prob(h0[t][1])
                if self.entropy_reg:
                    loss_policy += self.entropy_factor * -torch.sum([p * torch.log(p) for p in h0[t][3].probs][0])
                loss_value += torch.square(Q_n.detach() - v_pred)
        loss_policy /= self.M
        loss_value /= self.M
        reward /= self.M

        # compute the epoch gradient and update policy head weights 
        self.train(self.model, loss_policy, self.optimizer)

        # compute the epoch gradient and update the value head weights 
        self.train(self.val_fun, loss_value, self.optim_value)
        
        #return traces average loss and reward
        return loss_policy.item(), loss_value.item(), reward