import time
import torch
import gym
import argparse
from numpy import mean, std
from Model import *

# TODO everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-run_name", action="store", type=str, default=None)
    parser.add_argument('-render', action='store_true')
    parser.add_argument('-quantum', action='store_true')
    parser.add_argument('-device', action='store', type=str, default="cuda")
    args = parser.parse_args()

    mlp_policy = MLP(4, 2, quantum=args.quantum).to(args.device)

    print(f"Loading weights from of {args.run_name}")
    try:
        state_dict = torch.load(f"exp_results/{args.run_name}_weights.pt")
        mlp_policy.load_state_dict(state_dict)
    except Exception as e:
        exit(f"Couldn't load the checkpoint at {args.run_name}_weights.pt: {e}")
    
    env = gym.make('CartPole-v1')
    s = env.reset()
    env.render()
    input("Press enter to start the evaluation...")

    # test an evaluation run after the model is done training
    trials = 100
    ts_ep = [0]*trials
    for i in range(trials):
        done = False
        s = env.reset()
        if args.render:
            env.render()
        while not done:
            with torch.no_grad():
                mlp_policy.eval()
                s_next, _, done, _ = env.step(int(argmax(mlp_policy.forward(s, args.device))))
                s = s_next
            ts_ep[i] += 1
            if args.render:
                env.render()
        print(f"[{i}] {ts_ep[i]} steps")
    print(f"Average steps over {trials} trials: {mean(ts_ep)} +- {std(ts_ep)}")

    env.close()

if __name__ == "__main__":
    main()