import csv
import matplotlib.pyplot as plt
from operator import sub
from matplotlib.ticker import FormatStrFormatter


def plot_curves(filename, category, epoch_min, target):
    with open(filename) as csvfile:
        r = csv.DictReader(csvfile)
        epochs = {False: [], True: []}
        errors = {False: [], True: []}

        for row in r:
            e = int(float(row["epochs"]))
            if e >= epoch_min:
                epochs[bool(int(row["encoder"]))].append(e)
                errors[bool(int(row["encoder"]))].append(float(row[category]))

    barWidth = 0.275

    fig = plt.gcf()
    #fig.set_size_inches(4,3)

    #formatter = ScalarFormatter()
    #formatter.set_scientific(False)
    #ax = plt.gca()
    #ax.yaxis.set_major_formatter(formatter)
    #ax.set_aspect(2)

    indices = list(range(len(errors[False])))

    plt.bar(indices, errors[False], width=barWidth, edgecolor='white', label="Random initialization")
    plt.bar([x+0.3 for x in indices], errors[True], width=barWidth, edgecolor='white', label="With initializer")
    plt.xticks([r + 0.15 for r in indices], [epochs[False][i] for i in indices], rotation=0)

    if epoch_min > 0:
        ylim = list(plt.ylim())
        ylim[0] = 0.9925*min(errors[True]+errors[False])
        ylim[1] = 1.0025*max(errors[True]+errors[False])
        plt.ylim(*ylim)

    #o = 0.0
    #for a,b in bars.items():
    #    r = plt.bar(np.arange(o, len(b)), b, width=barWidth, edgecolor='white', label=a)
    #    o += 0.2

    # Add xticks on the middle of the group bars
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('IOD normalized RMSE in %')
    #plt.xticks([r + 0.15 for r in range(len(list(bars.values())[0]))], [labelmap[c] for c in data.keys() if c != key], rotation=25)

    #autolabel(r, plt.gca())
    #autolabel(rect2, plt.gca())


    plt.legend()
    ax = plt.gca()
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%02d'))

    target_ratio = 1
    current_aspect = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    ax.set_aspect(target_ratio/current_aspect)


    #plt.subplots_adjust(bottom=0.15)
    #plt.show()
    plt.savefig(target, bbox_inches='tight')
    plt.clf()

plot_curves("/tmp/extracted_encoder_epochs.csv", "h68", 0, "/tmp/initializer_h68_all_epochs.png")
plot_curves("/tmp/extracted_encoder_epochs.csv", "e68", 0, "/tmp/initializer_e68_all_epochs.png")
#plot_curves("/tmp/extracted_encoder_epochs.csv", "h49", 0, "/tmp/initializer_h49_all_epochs.png")
#plot_curves("/tmp/extracted_encoder_epochs.csv", "e49", 0, "/tmp/initializer_e49_all_epochs.png")
plot_curves("/tmp/extracted_encoder_epochs.csv", "h68", 50, "/tmp/initializer_h68_high_epochs.png")
plot_curves("/tmp/extracted_encoder_epochs.csv", "e68", 50, "/tmp/initializer_e68_high_epochs.png")
#plot_curves("/tmp/extracted_encoder_epochs.csv", "h49", 50, "/tmp/initializer_h49_high_epochs.png")
#plot_curves("/tmp/extracted_encoder_epochs.csv", "e49", 50, "/tmp/initializer_e49_high_epochs.png")


plot_curves("/tmp/extracted_encoder_epochs_fixed_pair_no_varthresh.csv", "h68", 0, "/tmp/fixed/initializer_h68_all_epochs.png")
plot_curves("/tmp/extracted_encoder_epochs_fixed_pair_no_varthresh.csv", "e68", 0, "/tmp/fixed/initializer_e68_all_epochs.png")
plot_curves("/tmp/extracted_encoder_epochs_fixed_pair_no_varthresh.csv", "h68", 50, "/tmp/fixed/initializer_h68_high_epochs.png")
plot_curves("/tmp/extracted_encoder_epochs_fixed_pair_no_varthresh.csv", "e68", 50, "/tmp/fixed/initializer_e68_high_epochs.png")
