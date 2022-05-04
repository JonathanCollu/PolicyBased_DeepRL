from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


class LearningCurvePlot:
    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch")
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