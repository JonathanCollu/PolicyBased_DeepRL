import time
import argparse
import torch
import gym
from utils import *
from Model import *
from Algorithms.Reinforce import Reinforce
from Algorithms.AC_bootstrap import ACBootstrap

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

    # parse algorithm parameters
    parser.add_argument('-alg', action='store', type=str, default='reinforce')
    parser.add_argument('-traces', action='store', type=int, default=5)
    parser.add_argument('-trace_len', action='store', type=int, default=500)
    parser.add_argument('-epochs', action='store', type=int, default=1000)
    parser.add_argument('-n', action='store', type=int, default=10)
    parser.add_argument('-gamma', action='store', type=float, default=0.99)
    parser.add_argument('-baseline', action='store_true')
    parser.add_argument('-entropy', action='store_true')
    parser.add_argument('-entropy_factor', action='store', type=float, default=0.2)
    parser.add_argument('-use_es', action='store', type=int, default=None)
    args = parser.parse_args()

    optimizers = {  'adam': torch.optim.Adam,
                    'sgd': torch.optim.SGD,
                    'rms': torch.optim.RMSprop
                }

    env = gym.make("CartPole-v1")

    run_name = "exp_results/" + args.run_name
    optimum = 500

    n_repetitions = 5

    # instantiate plot and add optimum line
    plot = LearningCurvePlot(title = args.alg.upper())
    plot.add_hline(optimum, label="optimum")
    
    # run algorithm for n_repetitions times
    curve = None
    for i in range(n_repetitions):
        # instantiate models
        mlp_policy = MLP(4, 2, False, quantum=args.quantum)
        opt_policy = optimizers[args.optimizer](mlp_policy.parameters(), args.optim_lr)
        if args.baseline or args.alg == "AC_bootstrap":
            mlp_value = MLP(4, 2, True)
            opt_value = optimizers[args.optimizer_v](mlp_value.parameters(), args.optim_lr_v)
        else:
            mlp_value = None
            opt_value = None

        # instantiate algorithm
        if args.alg == "reinforce":
            alg = Reinforce(env, mlp_policy, opt_policy, mlp_value, opt_value, args.epochs, args.traces, args.gamma,
                args.entropy, args.entropy_factor, args.use_es, run_name+str(i), args.device)
        elif args.alg == "AC_bootstrap":
            alg = ACBootstrap(env, mlp_policy, opt_policy, mlp_value, opt_value, args.epochs, args.traces, args.trace_len,
                args.n, args.baseline, args.entropy, args.entropy_factor, args.use_es, run_name+str(i), args.device)
        else:
            print("Please select a valid model")
        
        # run and add rewards to curve
        now = time.time()
        rewards = np.array(alg())
        print("Running one setting takes {} minutes".format(round((time.time()-now)/60, 2)))    
        if curve is None:
            curve = rewards
        else:
            curve += rewards
            
    # average curve over repetitions
    curve /= n_repetitions
    
    # smooth curve of average rewards and add to plot
    curve = smooth(curve, 35, 1)
    plot.add_curve(curve, label=r"label")

    # save plot
    plot.save("plots/" + args.run_name + ".png")

    if args.evaluate:
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
            time.sleep(0.1)

    env.close()

if __name__ == "__main__":
    main()