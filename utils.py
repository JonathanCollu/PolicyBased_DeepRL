from copy import deepcopy
import numpy as np
import time
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from Algorithms.Reinforce import Reinforce
from Algorithms.AC_bootstrap import ACBootstrap


class LearningCurvePlot:
    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Reward")
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
            label: string to appear as label in plot legend
        '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls="--",c="k",label=label)

    def save(self,name="test.png"):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)


def smooth(y, window, poly=1):
    ''' y: vector to be smoothed 
        window: size of the smoothing window
    '''
    return savgol_filter(y,window,poly)

def average_over_repetitions(
        algorithm,
        env,
        model,
        optimizer,
        epochs=100,
        M=5,
        T=500,
        gamma=0.9,
        sigma=None,
        n=5,
        baseline_sub = True,
        n_repetitions=10,
        smoothing_window=51):

    reward_results = np.empty([n_repetitions, epochs]) # Result array
    now = time.time()
    copy = deepcopy(model)
    for rep in range(n_repetitions): # Loop over repetitions
        if algorithm == "reinforce":
            alg = Reinforce(env, model, optimizer, epochs, M, gamma, sigma, baseline_sub)
        elif algorithm == "AC_bootstrap":
            alg = ACBootstrap(env, model, optimizer, epochs, M, T, n, sigma, baseline_sub)
        model = deepcopy(copy)

        reward_results[rep] = alg()
        
    print("Running one setting takes {} minutes".format((time.time()-now)/60))
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve