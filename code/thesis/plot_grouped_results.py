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

    x = data[key]
    y = [float(v) for v in data[split]]

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.plot(x, y, '--ro', c=color_map[split])

    fig = plt.gcf()
    fig.set_size_inches(4,2.5)  #width,height
    plt.xticks(x)
    plt.xlabel(key)
    plt.ylabel("IOD normalized RMSE in %")
    plt.grid()

    target_ratio = 1
    current_aspect = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    #print(current_aspect, 1/current_aspect)
    #print(sub(*ax.get_ylim()) / sub(*ax.get_xlim()))
    ax.set_aspect(target_ratio/current_aspect)
    #print(sub(*ax.get_ylim()) / sub(*ax.get_xlim()))

    #plt.legend()

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
    if not only_main:
        plot_curve(filename, key, "e", target)
        plot_curve(filename, key, "h", target)
        plot_curve(filename, key, "_49", target)
        plot_curve(filename, key, "_68", target)
        plot_curve(filename, key, "_all", target)


#plot_curves("/home/simon/Desktop/masters-thesis/results/extracted_loss_functions.csv", "loss_function")


if True:
    target = "/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_architecture/plots"
    plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_architecture/n_hgs_cmp.csv", "n_hgs", target, only_main=True)
    plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_architecture/n_lm_cmp.csv", "n_lm", target, only_main=True)
    plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_architecture/n_res_modules_cmp.csv", "n_res_modules", target, only_main=True)
    plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_architecture/n_features_cmp.csv", "n_features", target, only_main=True)
    plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_architecture/hg_depth_cmp.csv", "hg_depth", target, only_main=True)

target = "/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_other/plots"
plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_other/extracted_jensen_shannon.csv", "heatmap_sigma", target, only_main=True)
plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_other/extracted_rotation_all.csv", "angle", target, only_main=True)
plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_other/extracted_pdw_30deg.csv", "pd_weight", target, only_main=False)