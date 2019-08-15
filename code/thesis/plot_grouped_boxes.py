import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict
import numpy as np
from operator import sub


def plot_curves(filename, key, labelmap, barWidth=0.15, target_ratio=1,ymin=1.75,ymax=4.75, figsize=None):
    def autolabel(rects, ax):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.025*height, '%0.2f' % height, ha='center', va='bottom')

    whitelist = [key, "h68", "h49", "e49", "e68", "e", "h", "_49", "_68", "_all"]

    with open(filename) as csvfile:
        r = csv.DictReader(csvfile)
        data = defaultdict(list)

        for row in r:
            for col,val in row.items():
                if col in whitelist:
                    data[col].append(val)

    bars = {}
    groups = data[key]
    for i,group in enumerate(groups):
        bars[group] = []
        for k,v in data.items():
            if k != key:
                bars[group].append(float(v[i]))

    fig = plt.gcf()
    if figsize is not None:
        fig.set_size_inches(*figsize)
    plt.ylim(ymin,ymax)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)

    print(bars)

    o = 0.0
    for a,b in bars.items():
        r = plt.bar(np.arange(o, len(b)), b, width=barWidth, edgecolor='white', label=a)
        o += barWidth*1.25

    # Add xticks on the middle of the group bars
    plt.xlabel('Category', fontweight='bold')
    plt.ylabel('IOD normalized RMSE in %')
    plt.xticks([r + barWidth for r in range(len(list(bars.values())[0]))], [labelmap[c] for c in data.keys() if c != key], rotation=25)

    #autolabel(r, plt.gca())
    #autolabel(rect2, plt.gca())

    current_aspect = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    ax.set_aspect(target_ratio/current_aspect)

    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("/tmp/%s.png" % key, bbox_inches='tight')
    #plt.show()
    plt.clf()

labelmap = {
    "e49" : "easy (49)",
    "h49" : "diff (49)",
    "e68" : "easy (68)",
    "h68" : "diff (68)",
    "e" : "easy avg.",
    "h" : "diff avg.",
    "_49" : "49 avg.",
    "_68" : "68 avg.",
    "_all" : "all avg.",
}
#plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_other/extracted_loss_functions.csv", "loss_function", labelmap)
plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_other/extracted_loss_normalization.csv", "normalize_loss", labelmap, target_ratio=1.5, barWidth=0.3, figsize=(8,9), ymax=4.6)
plot_curves("/home/simon/Desktop/masters-thesis/masters-thesis/results/hg_other/extracted_regression.csv", "regression", labelmap, target_ratio=0.75, barWidth=0.3,ymax=5.0, figsize=(5,5))
