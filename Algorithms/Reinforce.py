import torch
from Algorithms.PolicyBased import PolicyBased as PB


class Reinforce(PB):
    def __init__(
            self, env, model, optimizer, model_v, optimizer_v,
            epochs, M, gamma, entropy_reg, entropy_factor,
            use_es, run_name, device):

        self.env = env
        self.model = model
        self.model_v = model_v
        self.optimizer = optimizer
        self.optim_value = optimizer_v
        self.set_device(device)
        self.epochs = epochs
        self.M = M
        self.T = None # get full episodes
        self.gamma = gamma
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
            R = 0
            # len-2 reason: -1 for having 0..len-1 and -1 for skipping last state
            for t in range(len(h0) - 2, -1, -1):
                R = h0[t][2] + self.gamma * R
                if self.model_v is not None:
                    v = self.model_v.forward(h0[t][0], self.device)
                    loss_value += torch.square(R - v)
                    v = v.detach()
                else:
                    v = 0
                loss_policy += (R - v) * -h0[t][3].log_prob(h0[t][1])
                if self.entropy_reg:
                    loss_policy += self.entropy_factor * h0[t][3].entropy()
        loss_policy /= self.M
        loss_value /= self.M
        reward /= self.M
        
        return loss_policy, loss_value, reward
        
    def train_(self, loss_policy, loss_value, reward):
        # compute the epoch gradient and update weights
        self.train(self.model, loss_policy, self.optimizer)

        # if using baseline sub update value function model weights
        if self.model_v is not None:
            self.train(self.model_v, loss_value, self.optim_value) 
        
        #return traces average loss_policy and reward
        return loss_policy.item(), loss_value.item(), reward