# Policy-based Deep Reinforcement Learning
This repository contains the solution to the third assignment of the course Reinforcement Learning from Leiden Univeristy. It contains a framework for experimenting with different **policy-based reinforcement learning** techniques on OpenAI's Gym environments. In our specific case we succesfully applied **REINFORCE** and **Actor-Critic** algorithms (in various configurations) to the **"CartPole-V1"** environment. For a detailed description of the methodologies used and the experiments carried out, please refer to the <a href=https://github.com/JonathanCollu/RL_A3/blob/main/report_A3.pdf>full report</a>.

## Authors
<a href="https://github.com/OhGreat">Dimitrios Ieronymakis</a>, <a href="https://github.com/JonathanCollu">Jonathan Collu</a> and <a href="https://github.com/riccardomajellaro">Riccardo Majellaro</a>

## Requirements
To run the available scripts, a `Python 3` environment is required, together with the packages specified in the `requirements.txt` file, in the main directory. In order to install the requirements, run the following command on the `Python 3` environment:
 
 ```
 pip install -r requirements.txt
 ````

## How to train all the configurations
All the experiments presented in the report are fully reproducible by running the following command from the main folder of the repository:

```
./experiment/basic_exps.sh
``` 
Remember to change the script permissions in order to make it executable as a program.

Note: Windows users can convert the above file to a `.bat` by simply removing the shebang (first line), comments, converting "\\" to "^" and deleting ";".

## How to run a configuration (training)

```
python experiment.py
```
from the main directory, along with the following possible arguments:
- `-run_name`: name of your choice for the configuration.
- `-device`: where to execute the computations (e.g. "cpu" or "cuda").
- `-optimizer`: choose an optimizer between "adam", "sgd" and "rms" for the policy net.
- `-optim_lr`: learning rate of the optimizer.
- `-optimizer_v`: choose an optimizer between "adam", "sgd" and "rms" for the value net.
- `-optim_lr_v`: learning rate of the optimizer_v.
- `-quantum`: use a quantum layer as an output layer for the policy network.
- `-alg`: choose between "reinforce" and "AC_bootstrap".
- `-epochs`: number of epochs (i.e. updates).
- `-traces`: number of traces per epoch (averaged in a single update).
- `-trace_len`: length of a trace.
- `-n`: number of steps for bootstrapping.
- `-gamma`: discount factor.
- `-baseline`: to use baseline subtraction.
- `-entropy`: to use entropy regularization.
- `-entropy_factor`: entropy regularization factor.
- `-use_es`: set to 0 or 1 to use evolutionary strategies as described in the report.

## How to evaluate a configuration
Run the command below from the main directory
```
python evaluate.py
```
along with the following arguments:
- `-run_name`, name of your choice for the configuration.
- `-render`, to visualize the environment.
- `-device`, to indicate where to execute the computations (e.g. "cpu" or "cuda").
- `-quantum`, to use a quantum layer as an output layer for the policy network.
