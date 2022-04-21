import torch
import gym
from Algorithms.REINFORCE import REINFORCE
from Model import MLP

env = gym.make('CartPole-v1')
mlp = MLP(4,2)
opt = torch.optim.SGD(mlp.parameters(), lr = 0.001)

reinforce = REINFORCE(env, mlp, opt)

reinforce()
print(0)