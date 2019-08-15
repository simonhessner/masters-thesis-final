import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_curve(filename, key, split, lines=True, add_filename=""):
    with open(filename) as csvfile:
        r = csv.DictReader(csvfile)
        x = []
        e49 = []
        h49 = []
        e68 = []
        h68 = []
        depth = []
        for row in r:
            try:
                x.append(int(row[key]))
            except ValueError:
                x.append(row[key])

            if "MIN(best_e49)" in row:
                e49.append(float(row["MIN(best_e49)"]))
                h49.append(float(row["MIN(best_h49)"]))
                e68.append(float(row["MIN(best_e68)"]))
                h68.append(float(row["MIN(best_h68)"]))
            else:
                e49.append(float(row["best_e49"]))
                h49.append(float(row["best_h49"]))
                e68.append(float(row["best_e68"]))
                h68.append(float(row["best_h68"]))
            if "hg_depth" in row:
                depth.append(int(row["hg_depth"]))

    print("test")

    if split == "e49":
        if lines:
            plt.plot(x, e49, '--ro', label="easy without outline (49 landmarks)")
        else:
            plt.scatter(x, e49, s=10, c=["r" if d < 5 else "black" for d in depth], label="easy without outline (49 landmarks)")
    if split == "h49":
        if lines:
            plt.plot(x, h49, '--bo', label="difficult without outline (49 landmarks)")
        else:
            plt.scatter(x, h49, s=10, c=["b" if d < 5 else "black" for d in depth], label="difficult without outline (49 landmarks)")
    if split == "e68":
        if lines:
            plt.plot(x, e68, '--go', label="easy with outline (68 landmarks)")
        else:
            plt.scatter(x, e68, s=10, c=["g" if d < 5 else "black" for d in depth], label="easy with outline (68 landmarks)")
    if split == "h68":
        if lines:
            plt.plot(x, h68, '--mo', label="difficult with outline (68 landmarks)")
        else:
            plt.scatter(x, h68, s=10, c=["m" if d < 5 else "black" for d in depth], label="difficult with outline (68 landmarks)")

    #fig = plt.gcf()
    #fig.set_size_inches(16,16)
    if lines:
        plt.xticks(x)
    plt.xlabel(key)
    plt.ylabel("IOD normalized RMSE")
    plt.legend()
    plt.savefig("/tmp/%s%s_%s.png" % (key, add_filename, split), bbox_inches='tight')
    #plt.show()
    plt.clf()


def plot_curves(filename, key, lines=True, add_filename=""):
    plot_curve(filename, key, "e49", lines, add_filename)
    plot_curve(filename, key, "h49", lines, add_filename)
    plot_curve(filename, key, "e68", lines, add_filename)
    plot_curve(filename, key, "h68", lines, add_filename)


def plot_pdm_layers(filename, target):
    with open(filename) as csvfile:
        r = csv.DictReader(csvfile)

        points = []

        for row in r:
            points.append((int("".join([a for a in row["layers"] if a.isdigit()])), float(row["e"])))

    points = sorted(points, key=lambda x: x[0])

    x = [e[0] for e in points]
    y = [e[1] for e in points]

    cutoff = 0

    plt.plot(x[cutoff:], y[cutoff:], '--ro', label="error on 300-W test set")
    plt.xlabel("Neurons")
    plt.xticks([1, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024][cutoff:], rotation=90)
    plt.ylabel("IOD normalized RMSE")
    plt.yscale("log")
    plt.yticks([min(y), 1.0, 5.0, 10.0, 20.0, max(y)])

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)

    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(target, bbox_inches='tight')
    #plt.show()
    plt.clf()


def plot_regression(filename):
    def autolabel(rects, ax):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.025*height, '%0.2f' % height, ha='center', va='bottom')

    with open(filename) as csvfile:
        r = csv.DictReader(csvfile)

        e49 = {}
        h49 = {}
        e68 = {}
        h68 = {}

        for row in r:
            e49[row["regression"]] = float(row["MIN(best_e49)"])
            h49[row["regression"]] = float(row["MIN(best_h49)"])
            e68[row["regression"]] = float(row["MIN(best_e68)"])
            h68[row["regression"]] = float(row["MIN(best_h68)"])

    barWidth = 0.3

    # set height of bar
    bars1 = [x["dsnt"] for x in [e49, e68, h49, h68]]
    bars2 = [x["linear"] for x in [e49, e68, h49, h68]]

    import numpy as np
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth*1.1 for x in r1]

    #plt.yticks(bars1 + bars2 + [1, 8])
    #plt.margins(0.075,)
    plt.ylim(0,6)


    # Make the plot
    rect1 = plt.bar(r1, bars1, color='r', width=barWidth, edgecolor='white', label='Linear')
    rect2 = plt.bar(r2, bars2, color='b', width=barWidth, edgecolor='white', label='DSNT')

    # Add xticks on the middle of the group bars
    plt.xlabel('Category', fontweight='bold')
    plt.ylabel('IOD normalized RMSE')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['easy (49)', 'easy (68)', 'diff. (49)', ' diff. (68)'])

    autolabel(rect1, plt.gca())
    autolabel(rect2, plt.gca())

    # Create legend & Show graphic
    plt.legend()
    #plt.show()
    plt.savefig("/tmp/hg_regression.png")

plot_curves("/tmp/extracted_loss.csv", "loss_function")

plot_regression("/tmp/extracted_regression.csv")

exit()

plot_pdm_layers("/tmp/thesis/300w_layers_68.csv", "/tmp/300w_layers_68.png")
plot_pdm_layers("/tmp/thesis/300w_multipie_layers_68.csv", "/tmp/300w_multipie_layers_68.png")
plot_pdm_layers("/tmp/thesis/300w_multipie_layers_49.csv", "/tmp/300w_multipie_layers_49.png")

exit()

plot_curves("/tmp/thesis/_extracted/hg_depth_cmp.csv", "hg_depth")
plot_curves("/tmp/thesis/_extracted/n_lm_cmp.csv", "n_lm")
plot_curves("/tmp/thesis/_extracted/n_hgs_cmp.csv", "n_hgs")
#plot_curves("/tmp/thesis/_extracted/n_hgs_all_cmp.csv", "n_hgs", lines=False, add_filename="_all")
plot_curves("/tmp/thesis/_extracted/n_res_modules_cmp.csv", "n_res_modules")
plot_curves("/tmp/thesis/_extracted/n_features_cmp.csv", "n_features")
#plot_curves("/tmp/thesis/_extracted/trainable_parameters_depth_5_cmp.csv", "trainable_parameters", lines=False, add_filename="_depth_5")
plot_curves("/tmp/thesis/_extracted/trainable_parameters_except_depth_5_cmp.csv", "trainable_parameters", lines=False, add_filename="_except_depth_5")
plot_curves("/tmp/thesis/_extracted/trainable_parameters_all_depths_cmp.csv", "trainable_parameters", lines=False, add_filename="_all_depths")
plot_curves("/tmp/thesis/_extracted/layers.csv", "layers")