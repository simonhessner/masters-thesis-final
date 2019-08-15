import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict
from operator import sub
from common.util import mkdir_if_not_exists
import os


def plot_curve(filename, key, split, target):
    color_map = {
        "e49" : "red",
        "h49" : "blue",
        "e68" : "green",
        "h68" : "purple",
        "e": None,
        "h": None,
        "_49": None,
        "_68": None,
        "_all": None
    }

    with open(filename) as csvfile:
        r = csv.DictReader(csvfile)
        data = defaultdict(list)

        for row in r:
            for col,val in row.items():
                data[col].append(val)

    x = [int(x) for x in data[key]]
    y = [float(v) for v in data[split]]
    depths = [int(x) for x in data["hg_depth"]]

    x1 = [x[i] for i in range(len(x)) if depths[i] < 5]
    x2 = [x[i] for i in range(len(x)) if depths[i] == 5]

    y1 = [y[i] for i in range(len(y)) if depths[i] < 5]
    y2 = [y[i] for i in range(len(y)) if depths[i] == 5]

    fig = plt.gcf()
    fig.set_size_inches(5.5,3.5)  #width,height

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    plt.scatter(x2, y2, s=10, c="black", label="hg_depth = 5")
    plt.scatter(x1, y1, s=10, c=color_map[split], label="hg_depth < 5")

    plt.xlabel(key.replace("_", " "))
    plt.ylabel("IOD normalized RMSE in %")

    target_ratio = 1
    current_aspect = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    #print(current_aspect, 1/current_aspect)
    #print(sub(*ax.get_ylim()) / sub(*ax.get_xlim()))
    ax.set_aspect(target_ratio/current_aspect)
    #print(sub(*ax.get_ylim()) / sub(*ax.get_xlim()))
    plt.legend()

    targetdir = os.path.join(target, key)
    mkdir_if_not_exists(targetdir)
    plt.savefig(os.path.join(targetdir, "%s_%s.png" % (key, split)), bbox_inches='tight')

    #plt.show()
    plt.clf()


def plot_curves(filename, key, target, only_main=False):
    plot_curve(filename, key, "e49", target)
    plot_curve(filename, key, "h49", target)
    plot_curve(filename, key, "e68", target)
    plot_curve(filename, key, "h68", target)

target = "/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_architecture/plots"
plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_architecture/trainable_parameters_all_depths_cmp.csv", "trainable_parameters", target, only_main=True)