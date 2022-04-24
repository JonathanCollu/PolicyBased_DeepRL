import torch
import gym
from utils import *
from Model import *



def main():
    
    n_repetitions = 10
    smoothing_window = 3

    env = gym.make('CartPole-v1')
    mlp = MLPHybrid(4,2, Ansatz_mode=False) #MLP(4,2)
    opt = torch.optim.Adam(mlp.parameters(), lr = 0.001)

    algorithm = 'reinforce' # 'AC_bootstrap', 'reinforce' 
    epochs = 1000
    M = 5
    T = 500
    gamma = 0.9
    sigma = None

    optimum = 500

    Plot = LearningCurvePlot(title = 'REINFORCE')  

    l_c = average_over_repetitions(algorithm, env, mlp, opt, epochs=epochs, M=M, T=T, gamma=gamma, sigma=sigma, n_repetitions=n_repetitions, smoothing_window=smoothing_window)
    Plot.add_curve(l_c,label=r'label')
    Plot.add_hline(optimum, label="optimum")
    Plot.save('Reinforce.png')
    print(0)

if __name__ == "__main__":
    main()