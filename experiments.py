import torch
import gym
from utils import *
from Model import *


def main():
    n_repetitions = 10
    smoothing_window = 3

    env = gym.make("CartPole-v1")
    mlp = MLP(4,2, quantum=True)
    opt = torch.optim.Adam(mlp.parameters(), lr = 0.001)

    algorithm = "reinforce" # "AC_bootstrap", "reinforce"
    epochs = 1000
    M = 5
    T = 500
    n = 10
    gamma = 0.9
    baseline_sub = True
    entropy_reg = True
    entropy_factor = 0.2

    optimum = 500

    Plot = LearningCurvePlot(title = algorithm.upper())  

    l_c = average_over_repetitions(
        algorithm, env, mlp, opt, epochs=epochs,
        M=M, T=T, gamma=gamma, n=n, baseline_sub=baseline_sub,
        entropy_reg=entropy_reg, entropy_factor=entropy_factor,
        n_repetitions=n_repetitions, smoothing_window=smoothing_window)
    Plot.add_curve(l_c,label=r'label')
    Plot.add_hline(optimum, label="optimum")
    Plot.save(algorithm + ".png")

if __name__ == "__main__":
    main()