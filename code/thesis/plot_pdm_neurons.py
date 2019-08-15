import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_curve(filename):
    with open(filename) as csvfile:
        r = csv.DictReader(csvfile)

        layers = []
        #errors_49 = []
        errors_68 = []

        for row in r:
            _layers = row["layers"].replace("[", "").replace("]","")
            if "," in _layers:
                continue
            neurons = int(_layers)
            if neurons < 48:
                continue
            #if neurons not in [2**i for i in range(12)]:
            #    continue
            layers.append(neurons)
            #errors_49.append(float(row["test_error_49"]))
            #errors_68.append(float(row["test_error_68"]))
            errors_68.append((float(row["best_e68"])+float(row["best_h68"]))/2)

    combined = list(zip(layers,errors_68))
    combined = sorted(combined, key=lambda x: x[0])

    layers = [x[0] for x in combined]
    #errors_49 = [x[1] for x in combined]
    errors_68 = [x[1] for x in combined]

    ax = plt.gca()
    #ax.set_aspect(2)

    plt.xticks([1,32,64,128,192,256,384,512], rotation=0)
    ax.set_yscale("log", basey=10)
    ax.set_yticks([min(errors_68),0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0,max(errors_68)])
    plt.minorticks_off()
    #plt.grid()

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    #ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    #plt.plot(layers, errors_49, '--ro', label="test error (49 landmarks)")
    plt.plot(layers, errors_68, '--bo', label="test error (68 landmarks)")

    #plt.legend()
    #plt.subplots_adjust(bottom=0.15)

    plt.xlabel("latent vector dimension")
    plt.ylabel("IOD normalized RMSE in % (log scale)")
    #plt.savefig("/tmp/pdm_reconstruction_error.png", bbox_inches='tight')
    plt.show()
    plt.clf()

plot_curve("/home/simon/Desktop/masters-thesis/masters-thesis/results/PDM/latent/extracted_reconstruction_error_latent_dim.csv")

