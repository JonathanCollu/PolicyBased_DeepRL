from utils import *

plot_name = "ac_n"
plot_title = "ActorCritic with different depth"

optimum = 500
repetitions = 3

Plot = LearningCurvePlot(title = plot_title)

curve = None
for name in ["ac_n50", "ac_n100", "ac_n200"]:
    c = np.load("exp_results/" + name + ".npy")[2]
    if curve is None:
        curve = c
    else:
        curve += c
curve /= repetitions
curve = smooth(curve, 35, 1)
Plot.add_curve(curve, label=r"label")

Plot.add_hline(optimum, label="optimum")

Plot.save("plots/" + plot_name + ".png")