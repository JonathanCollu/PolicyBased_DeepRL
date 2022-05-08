from utils import *
import numpy as np

plot_name = "reinforce"
plot_title = "REINFORCE and ActorCritic without bootstrap"

optimum = 500
repetitions = 3
run_names = [("reinf", "REINFORCE"), ("reinf_baseline", "ActorCritic + baseline sub."), ("reinforce_entr0.2", "REINFORCE + entropy reg."), ("reinforce_baseline_entr0.2", "ActorCritic + baseline sub. + entropy reg.")]


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