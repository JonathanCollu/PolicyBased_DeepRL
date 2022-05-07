import os
import gym
import torch
import argparse
from numpy import mean, std
from Model import *

def main():
    """ Evaluates all the weights found in the exp_results folder
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-run_name", action="store", type=str, default=None)
    parser.add_argument('-render', action='store_true')
    parser.add_argument('-quantum', action='store_true')
    parser.add_argument('-device', action='store', type=str, default="cuda")
    parser.add_argument('-out_file', action='store', type=str, default="all_exp_results.txt")
    args = parser.parse_args()

    # create environment
    env = gym.make('CartPole-v1')

    # reset out file
    f = open(args.out_file, "w")

    for filename in os.listdir("exp_results"):
        if filename.endswith(".pt"):
            try:
                if "quantum" in filename:
                    mlp_policy = MLP(4, 2, quantum=True).to(args.device)
                else:
                    mlp_policy = MLP(4, 2).to(args.device)
                state_dict = torch.load(f"exp_results/{filename}")
                mlp_policy.load_state_dict(state_dict)

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


                # writo to file
                out_str = f"Weights: {filename} | Reps: {trials} | trials: {mean(ts_ep)} +- {std(ts_ep)}\n"
                f = open(args.out_file, "a")
                f.write(out_str)
                f.close()
                print(out_str)
            except Exception as e:
                print(f"Couldn't load the checkpoint at {filename}: {e}")

    env.close()

if __name__ == "__main__":
    main()