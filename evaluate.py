import os
import torch
import gym
from Algorithms.AC_bootstrap import ACBootstrap
from Algorithms.Reinforce import Reinforce
import argparse
from numpy import mean, std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", action="store", type=str, default=None)
    parser.add_argument("-run_name", action="store", type=str, default=None)
    parser.add_argument('-render', action='store_true')
    parser.add_argument('-device', action='store', type=str, default="cuda")
    args = parser.parse_args()

    mlp_policy = MLP(4, 2, quantum=args.quantum)
    mlp_value = deepcopy(mlp_policy)

    print(f"Loading weights from of {args.run_name}")
    try:
        state_dict = torch.load(f"exp_results/{args.run_name}_weights.pt")
        net.load_state_dict(state_dict)
    except Exception as e:
        exit(f"Couldn't load the checkpoint at {args.run_name}_weights.pt: {e}")
    
    env = gym.make('CartPole-v1')

    # instantiate dql
    dql = DQL(
        device=args.device, env=env, input_is_img=args.use_img,
        model=net, loss=torch.nn.MSELoss, optimizer=torch.optim.Adam  # just to instantiate dql
    )

    # test an evaluation run after the model is done training
    trials = 200
    ts_ep = [0]*trials
    for i in range(trials):
        done = False
        s = env.reset()
        if args.render:
            env.render()
        if args.use_img:
            frames_mem = deque(maxlen=4)
            s = dql.collect_frame(frames_mem)
        while not done:
            with torch.no_grad():
                net.eval()
                s_next, _, done, _ = env.step(int(argmax(net.forward(torch.tensor(s, dtype=torch.float32, device=dql.device).unsqueeze(0)))))
                if args.use_img:
                    s_next = dql.collect_frame(frames_mem)
                s = s_next
            ts_ep[i] += 1
            if args.render:
                env.render()
        print(f"[{i}] {ts_ep[i]} steps")
    print(f"Average steps over {trials} trials: {mean(ts_ep)} +- {std(ts_ep)}")

    env.close()

if __name__ == "__main__":
    main()