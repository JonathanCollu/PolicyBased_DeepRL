from utils import *

optimum = 500
repetitions = 3

Plot = LearningCurvePlot(title = "test")

curve = None
for i in range(repetitions):
    c = np.load("exp_results/trial_" + str(i+1) + ".npy")[2]
    if curve is None:
        curve = c
    else:
        curve += c
curve /= repetitions
curve = smooth(curve, 35, 1)
Plot.add_curve(curve, label=r"label")

Plot.add_hline(optimum, label="optimum")

Plot.save("plots/" + "test" + ".png")