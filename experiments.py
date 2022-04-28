import argparse
import torch
import gym
from utils import *
from Model import *

def main():
    parser = argparse.ArgumentParser()
    
    # parse model parameters
    parser.add_argument('-evaluate', action='store_true')
    parser.add_argument('-run_name', action='store', type=str, default=None)
    parser.add_argument('-quantum', action='store_true')
    parser.add_argument('-optimizer', action='store', type=str, default='adam')
    parser.add_argument('-optim_lr', action='store', type=float, default=1e-3)
    parser.add_argument('-optimizer_v', action='store', type=str, default='adam')
    parser.add_argument('-optim_lr_v', action='store', type=float, default=1e-3)
    parser.add_argument('-device', action='store', type=str, default="cuda")

    # parse DQL parameters
    parser.add_argument('-alg', action='store', type=str, default='reinforce')
    parser.add_argument('-traces', action='store', type=int, default=5)
    parser.add_argument('-trace_len', action='store', type=int, default=500)
    parser.add_argument('-epochs', action='store', type=int, default=1000)
    parser.add_argument('-n', action='store', type=int, default=10)
    parser.add_argument('-gamma', action='store', type=float, default=0.99)
    parser.add_argument('-baseline', action='store_true')
    parser.add_argument('-entropy', action='store_true')
    parser.add_argument('-entropy_factor', action='store', type=float, default=0.2)
    args = parser.parse_args()

    optimizers = {  'adam': torch.optim.Adam,
                    'sgd': torch.optim.SGD,
                    'rms': torch.optim.RMSprop
                }

    n_repetitions = 1
    smoothing_window = 3

    env = gym.make("CartPole-v1")
    mlp_policy = MLP(4,2, quantum=args.quantum)
    mlp_value = deepcopy(mlp_policy)
    opt_policy = optimizers[args.optimizer](mlp_policy.parameters(), args.optim_lr)
    opt_value = optimizers[args.optimizer_v](mlp_value.parameters(), args.optim_lr_v)
    run_name = "exp_results/" + args.run_name

    optimum = 500

    Plot = LearningCurvePlot(title = args.alg.upper())  

    l_c = average_over_repetitions(
        args.alg, env, mlp_policy, opt_policy, epochs=args.epochs,
        M=args.traces, T=args.trace_len, gamma=args.gamma, n=args.n, 
        baseline_sub=args.baseline, entropy_reg=args.entropy, 
        entropy_factor=args.entropy_factor, val_fun=mlp_value, 
        optimizer_v = opt_value, run_name = run_name, device = args.device,
        n_repetitions=n_repetitions, smoothing_window=smoothing_window)
    Plot.add_curve(l_c,label=r'label')
    Plot.add_hline(optimum, label="optimum")
    Plot.save("plots/" + args.alg + ".png")

    if args.evaluate:
        from time import sleep
        done = False
        s = env.reset()
        env.render()
        input("Press enter to start the evaluation...")
        while not done:
            with torch.no_grad():
                mlp_policy.eval()
                s_next, _, done, _ = env.step(int(argmax(mlp_policy.forward(s, args.device))))
                s = s_next
            env.render()
            sleep(0.1)

    env.close()

if __name__ == "__main__":
    main()