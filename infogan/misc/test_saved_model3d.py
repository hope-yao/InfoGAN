
from viz import plot_cube
from viz import viz3
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


def plot_gen(imgs):
    ################################################################################
    # plt.figure(figsize=(10,10))
    # gs = gridspec.GridSpec(10, 10)
    # ax = []
    # for i in range(10):
    #     for j in range(10):
    #         ax = ax + [plt.subplot(gs[i,j], projection='3d')]
    #         ax[10*i+j].axis('equal')
    #         ax[10*i+j].axis('equal')
    #         x, y, z, t = viz3(numpy.squeeze(imgs[i][j]), 0)
    #         ax[10*i+j].scatter(x, y, z, c=t, marker='o', s=4)
    #
    i = j = 0
    from visualization.python.util_vtk import visualization

    img = numpy.squeeze(imgs[i][j])
    visualization(img, 0.62,  uniform_size=0.9)

    plt.show(True)