import pickle
import sys

with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)

    # import pptk
    # v = pptk.viewer(l3d_te[100].detach().cpu().numpy())
    # v.set(point_size=0.025)
    # v.wait()

    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D

    for i in range(10):
        if "l3d" in data and data["l3d"] is not None:
            fig = pyplot.figure()
            ax = Axes3D(fig)
            ax.scatter(data["l3d"][i, :, 0], data["l3d"][i, :, 1], data["l3d"][i, :, 2]) # TODO invert some axis??
            pyplot.savefig("/tmp/test_%d_3d.png" % i)
            pyplot.clf()
        else:
            print("l3d is None")

        if "l2d" in data and data["l2d"] is not None:
            pyplot.scatter(data["l2d"][i, :, 0], data["l2d"][i, :, 1] * -1)
            pyplot.savefig("/tmp/test_%d_2d.png" % i)
            pyplot.clf()
        else:
            print("l2d is None")

        if "before_pdm" in data and data["before_pdm"] is not None:
            pyplot.scatter(data["before_pdm"][i, :, 0], data["before_pdm"][i, :, 1] * -1)
            pyplot.savefig("/tmp/test_before_%d_pdm.png" % i)
            pyplot.clf()
        else:
            print("before_pdm is None")

        if "gt" in data and data["gt"] is not None:
            pyplot.scatter(data["gt"][i, :, 0], data["gt"][i, :, 1] * -1)
            pyplot.savefig("/tmp/test_%d_gt.png" % i)
            pyplot.clf()
        else:
            print("gt is None")