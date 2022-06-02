import torch
from Algorithms.PolicyBased import PolicyBased as PB


class ACBootstrap(PB):
    """ Parameters:
            - env : Environment to train our model
            - model : differentiable parametrized policy (model in pytorch)
            - optimzer : policy network optimization algorithm (torch optimizer)
            - model_v : value network (model in pytorch)
            - optimzer_v : value network optimization algorithm (torch optimizer)
            - epochs : number of epochs 
            - M : number of traces per epoch
            - T : trace length
            - n : estimation depth
            - baseline_sub : use or not baseline subtraction
            - entropy_reg : use or not entropy regularization
            - entropy_factor : entropy factor
            - use_es : flag to handle the usage of evolutionary strategies
            - run_name : name of the run
            - device : cuda or cpu 
    """
    def __init__(
            self, env, model, optimizer, model_v, optimizer_v, 
            epochs, M, T, n, baseline_sub, entropy_reg, entropy_factor, 
            use_es, run_name, device):

        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.model_v = model_v
        self.optim_value = optimizer_v
        self.set_device(device)
        self.epochs = epochs
        self.M = M
        self.T = T
        self.n = n
        self.baseline_sub = baseline_sub
        self.entropy_reg = entropy_reg
        self.entropy_factor = entropy_factor
        self.use_es = use_es
        self.run_name = run_name

    def epoch(self):
        loss_policy = torch.tensor([0], dtype=torch.float64, device=self.device) 
        loss_value = torch.tensor([0], dtype=torch.float64, device=self.device)
        reward = 0
        for _ in range(self.M):
            s = self.env.reset()
            h0, reward_t = self.sample_trace(s)
            reward += reward_t
            for t in range(len(h0) - 1): # -1 for skipping last state
                n = min(self.n, (len(h0) - 1 - t)) # to avoid indexes out of bound
                v = self.model_v.forward(h0[t + n][0], self.device)
                Q_n = sum([h0[t + k][2] for k in range(n)]) + v
                v_pred = self.model_v.forward(h0[t][0], self.device)
                if not self.baseline_sub:
                    loss_policy += Q_n.detach() * -h0[t][3].log_prob(h0[t][1])
                else:
                    loss_policy += (Q_n.detach() - v_pred.detach()) * -h0[t][3].log_prob(h0[t][1])
                if self.entropy_reg:
                    loss_policy -= self.entropy_factor * h0[t][3].entropy()
                loss_value += torch.square(Q_n.detach() - v_pred)
        loss_policy /= self.M
        loss_value /= self.M
        reward /= self.M
        
        return loss_policy, loss_value, reward

    def train_(self, loss_policy, loss_value, reward):
        # compute the epoch gradient and update policy head weights 
        self.train(self.model, loss_policy, self.optimizer)

        # compute the epoch gradient and update the value head weights 
        self.train(self.model_v, loss_value, self.optim_value)
        
        #return traces average loss and reward
        return loss_policy.item(), loss_value.item(), reward
