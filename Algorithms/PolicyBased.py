import os
from regex import F
import torch
import numpy as np
from torch.distributions import Categorical
from Model import argmax
from ES_base_framework.EA import EA
import ES_base_framework.Recombination as Recombination
import ES_base_framework.Mutation as Mutation
import ES_base_framework.Selection as Selection


class PolicyBased:
    """ Parameters:
            - optimzer : optimization algorithm (torch optimizer)
            - epochs : number of epochs 
            - M : number of traces per epoch
            - T : trace length
            - gamma : discount factor
            - model : differentiable parametrized policy (model in pytorch)
            - env : Environment to train our model 
    """
    def __init__(self, env, model, epochs, M, T, use_es, run_name, device):
        self.env = env
        self.epochs = epochs
        self.M = M
        self.T = T
        self.use_es = use_es
        self.run_name = run_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Computing on {self.device} device")
        self.model = model.to(device)

    def set_device(self, device):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Computing on {self.device} device")
        self.model.to(device)
        if self.model_v is not None:
            self.model_v.to(device)

    def __call__(self):
        rewards = []
        losses_p = []
        losses_v = []
        best_r_ep = 0
        best_avg = 0
        best_ep = 0
        # es configurations
        es_rec = Recombination.Discrete()
        es_mut = Mutation.IndividualSigma()
        es_sel = Selection.PlusSelection()
        es_eval = self.__class__.epoch
        # start training
        for i, epoch in enumerate(range(self.epochs)):
            if self.use_es and ((i+1) % 25) == 0:
                print("~~~~ Evolutionary Strategy Optimization ~~~~")
                # es on value layer
                es_value = EA(self, True, True, 100, 5, 40, 65, es_rec, es_mut, es_sel, es_eval, 1)
                new_weights, l_p, l_v, r = es_value.run() 
                for name, params in self.model.state_dict().items():
                    if "value_layer" in name:
                        if params.numel() == 1:
                            weights = torch.tensor(new_weights[-1:])
                        else:
                            weights = torch.tensor(new_weights[:-1].reshape((1, 64)))
                        self.model.load_state_dict({name: weights}, strict=False)
                # es on policy layer
                # es_policy = EA(self, False, True, 15, 1, 2, 130, es_rec, es_mut, es_sel, es_eval, 1)
                # new_weights, l_p, l_v, r = es_policy.run()
                # for name, params in self.model.state_dict().items():
                #     if "policy_layer" in name:
                #         if params.numel() == 2:
                #             weights = torch.tensor(new_weights[-2:])
                #         else:
                #             weights = torch.tensor(new_weights[:-2].reshape((2, 64)))
                #         self.model.load_state_dict({name: weights}, strict=False)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            else:
                l_p, l_v, r = self.train_(*self.epoch())
            losses_p.append(l_p)
            losses_v.append(l_v)
            rewards.append(r)
            print(f"[{epoch+1}] Epoch mean loss (policy): {round(l_p, 4)} | Epoch mean loss (value): {round(l_v, 4)} | Epoch mean reward: {r}")
            if rewards[-1] >= best_r_ep:
                best_r_ep = rewards[-1]
                print("New max number of steps in episode:", best_r_ep)
                if self.run_name is not None:
                    if best_r_ep == 500:
                        curr_avg = self.evaluate(25)
                        if curr_avg > best_avg:
                            save = True
                            best_avg = curr_avg
                        else: save = False
                    else: save = True
                    if save:
                        # remove old weights
                        if os.path.isfile(f"{self.run_name}_{best_ep}_weights.pt"): 
                            os.remove(f"{self.run_name}_{best_ep}_weights.pt")
                        # save model
                        torch.save(self.model.state_dict(), f"{self.run_name}_{epoch}_weights.pt")
                        best_ep = epoch
        if self.run_name is not None:
            # save steps per episode
            np.save(self.run_name, np.array([losses_p, losses_v, rewards]))
        return rewards

    def evaluate(self, trials):
        r_ep = [0]*trials
        for i in range(trials):
            done = False
            s = self.env.reset()
            while not done:
                with torch.no_grad():
                    self.model.eval()
                    pred = self.model.forward(s, self.device)
                    s_next, _, done, _ = self.env.step(int(argmax(pred)))
                    s = s_next
                r_ep[i] += 1
        return np.mean(r_ep)

    def select_action(self, s):
        # get the probability distribution of the actions
        dist = self.model.forward(s, self.device)

        # sample action from distribution
        dist = Categorical(dist)
        action = dist.sample()

        #return action and actions distribution
        return action, dist

    def sample_trace(self, s):
        reward = 0
        trace = []
        i = 0
        while True:
            if self.T is not None and i >= self.T:
                break
            i += 1
            a, a_dist = self.select_action(s)
            s_next, r, done, _ = self.env.step(a.item())
            trace.append((s, a, r, a_dist))
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