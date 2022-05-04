from utils import *
import numpy as np

plot_name = "ac_n"
plot_title = "ActorCritic with different n"

optimum = 500
repetitions = 3
run_names = [("ac_n50", "n=50"), ("ac_n125", "n=125"), ("ac_n250", "n=250")]

plot = LearningCurvePlot(title = plot_title)

for name, label in run_names:
    curve = None
    for i in range(repetitions):
        c = np.load(f"exp_results/{name}_{i}.npy")[2]
        if curve is None:
            curve = c
        else:
            curve += c
    curve /= repetitions
    curve = smooth(curve, 35, 1)
    plot.add_curve(curve, label=label)

plot.add_hline(optimum, label="optimum")

plot.save("plots/" + plot_name + ".png")