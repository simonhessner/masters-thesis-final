import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict
import numpy as np
from operator import  sub

def get_sort_value(layers):
    s = layers.replace("[", "").replace("]", "")
    l = [int(e.strip()) for e in s.split(",")]
    return l

def plot_curves(filename, split, labelmap, target):
    with open(filename) as csvfile:
        r = csv.DictReader(csvfile)
        data = defaultdict(list)

        for row in r:
            for col,val in row.items():
                data[col].append(val)

    x = data["layers"]
    y = [float(e) for e in data[split]]
    combined = list(zip(x,y))
    combined = sorted(combined, key=lambda e: get_sort_value(e[0]))
    #combined = sorted(combined, key=lambda e: e[1])
    x = [e[0] for e in combined]
    y = [e[1] for e in combined]

    latent_sizes = [get_sort_value(c)[0] for c in x]
    errs = {e : 1000 for e in latent_sizes}
    best = {}
    for i in range(len(x)):
        if y[i] < errs[latent_sizes[i]]:
            errs[latent_sizes[i]] = y[i]
            best[latent_sizes[i]] = x[i]

    x = list(errs.keys())
    y = list(errs.values())

    fig = plt.gcf()
    #fig.set_size_inches(3,6)

    plt.ylim(0.99*min(y), 1.0025*max(y))
    #plt.bar(x,y)
    for i in range(len(x)):
        #print(x[i], y[i])
        #plt.plot(x[i],[y[i]], "o", markersize=8, label=x[i])
        #plt.bar(x[i],[y[i]], label=x[i])
        plt.bar(i,[y[i]], label=best[x[i]], width=0.7)
    plt.legend()

    plt.xticks(list(range(len(x))), x)

    ax = plt.gca()
    target_ratio = 2.15
    current_aspect = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    ax.set_aspect(target_ratio/current_aspect)

    plt.xlabel("latent vector size")
    plt.ylabel("IOD normalized RMSE in %")
    #plt.grid()
    #plt.xticks(rotation=90)
    #plt.subplots_adjust(bottom=0.25)
    #ax.set_aspect(200)
    plt.savefig(target,  bbox_inches='tight')
    #plt.show()
    plt.clf()
    #exit()


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


plot_curves("/tmp/extracted_multilayer.csv", "e68", labelmap, "/tmp/pdm_multilayer_e68.png")
plot_curves("/tmp/extracted_multilayer.csv", "h68", labelmap, "/tmp/pdm_multilayer_h68.png")
plot_curves("/tmp/extracted_multilayer.csv", "e49", labelmap, "/tmp/pdm_multilayer_e49.png")
plot_curves("/tmp/extracted_multilayer.csv", "h49", labelmap, "/tmp/pdm_multilayer_h49.png")

