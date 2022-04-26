import torch
import gym
from utils import *
from Model import *


def main():
    n_repetitions = 10
    smoothing_window = 3

    env = gym.make("CartPole-v1")
    mlp = MLP(4,2) # MLPHybrid(4,2, Ansatz_mode=False)
    opt = torch.optim.Adam(mlp.parameters(), lr = 0.001)

    algorithm = "AC_bootstrap" # "AC_bootstrap", "reinforce"
    epochs = 1000
    M = 5
    T = 500
    n = 20
    gamma = 0.9
    sigma = None
    baseline_sub = True

    optimum = 500

    Plot = LearningCurvePlot(title = algorithm.upper())  

    l_c = average_over_repetitions(algorithm, env, mlp, opt, epochs=epochs, M=M, T=T, gamma=gamma, sigma=sigma, n=n, baseline_sub=baseline_sub, n_repetitions=n_repetitions, smoothing_window=smoothing_window)
    Plot.add_curve(l_c,label=r'label')
    Plot.add_hline(optimum, label="optimum")
    Plot.save(algorithm + ".png")

if __name__ == "__main__":
    main()