This repository contains the solution to the second assignment of the course Reinforcement Learning from Leiden Univeristy. It contains a framework for experimenting with a variety of **policy-based reinforcement learning** techniques to train agents that can solve OpenAI's environments. In our specific case we succesfully applied **REINFORCE** and **Actor-Critic** algorithms (in various configurations) to the **"CartPole-V1"** environment.

# Requirements
 To run the available scripts and algorithm configurations, a `Python 3` environment is required, together with the required packages, specified in the `requirements.txt` file, in the main directory. In order to install the requirements, run the following command from the main directory: 
 
 ```
 pip install -r requirements.txt
 ````

# How to train all the configurations

All the experiments presented in the report are fully repruducible by running the following command from the main folder of the repository:

```
./experiments/basic_exps.sh
``` 
.It is important to run the script out of the directory `experiments` using the command above to avoid errors with absoulute and relative paths. 
Furthermore, it is important to change the script permissions in order to make it executable as a program.

# How to train a configuration
`python experiment.py`
along with the arguments that are already used in `basic_exps.sh`
# How to evaluate a configuration
Run the command below from the main directory
`python evaluate.py`
along with the following arguments:
<ul>
<li>`-run_name`, name of the config to run </li>
<li>`-render`, to visualize the environment</li>
<li>`-device`, to indicate where to execute the computations (e.g. "cpu" or "cuda") </li>
<li>`-quantum`, to use a quantum layer as an output layer for the policy network</li>
</ul>
