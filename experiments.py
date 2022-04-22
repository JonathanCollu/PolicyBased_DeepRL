import torch
import gym
from Algorithms.REINFORCE import REINFORCE
from Algorithms.AC_bootstrap import ACBootstrap
from Model import MLP

env = gym.make('CartPole-v1')
mlp = MLP(4,2)
mlp_v = MLP(4, 2, v_func=True)
opt = torch.optim.SGD(mlp.parameters(), lr = 0.001)
opt_v = torch.optim.SGD(mlp_v.parameters(), lr = 0.001)

reinforce = REINFORCE(env, mlp, opt)
acb = ACBootstrap(env, mlp, opt, mlp_v, opt_v)

#reinforce()
acb()
print(0)