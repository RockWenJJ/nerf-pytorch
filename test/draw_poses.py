import matplotlib.pyplot as plt
import numpy as np
from load_blender import load_blender_data
from load_llff import load_llff_data
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    ''' Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.. This is one possible solution to Matplotlib's ax.set_aspect('euqal')
    and ax.axis('equal') not working for 3D

    Input:
      ax: a matplotlib axis, e.g., as output from plt.gca()
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# lego
# datadir = './data/nerf_synthetic/lego'
# imgs, poses, render_poses, hwf, i_split = load_blender_data(datadir, half_res=False, testskip=8)
# llff
datadir = './data/nerf_llff_data/fern'
images, poses, bds, render_poses, i_test = load_llff_data(datadir, 8, recenter=True, bd_factor=.75, spherify=False)

fig = plt.figure()
mycmap = plt.get_cmap('rainbow')
ax = fig.add_subplot(projection='3d')
for i, pose in enumerate(poses):
    x, y, z = pose[0][3], pose[1][3], pose[2][3]
    U = [pose[0][0], pose[1][0], pose[2][0]]
    V = [pose[0][1], pose[1][1], pose[2][1]]
    W = [pose[0][2], pose[1][2], pose[2][2]]
    # ax.quiver(x*3, y*3, z*3, U, V, W, colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.quiver(x, y, z, pose[0][0], pose[1][0], pose[2][0], colors=[1, 0, 0]) # red, x axis
    ax.quiver(x, y, z, pose[0][1], pose[1][1], pose[2][1], colors=[0, 1, 0]) # green, y axis
    ax.quiver(x, y, z, pose[0][2], pose[1][2], pose[2][2], colors=[0, 0, 1]) # blue, z axis
    # ax.text(x, y, z, '%d'%i)

# # rendered poses
# for i, pose in enumerate(render_poses):
#     x, y, z = pose[0][3], pose[1][3], pose[2][3]
#     U = [pose[0][0], pose[1][0], pose[2][0]]
#     V = [pose[0][1], pose[1][1], pose[2][1]]
#     W = [pose[0][2], pose[1][2], pose[2][2]]
#     # ax.quiver(x*3, y*3, z*3, U, V, W, colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     ax.quiver(x, y, z, pose[0][0], pose[1][0], pose[2][0], colors=[1, 0, 0])  # x axis
#     ax.quiver(x, y, z, pose[0][1], pose[1][1], pose[2][1], colors=[0, 1, 0])  # y axis
#     ax.quiver(x, y, z, pose[0][2], pose[1][2], pose[2][2], colors=[0, 0, 1])  # z axis

ax.scatter(0, 0, 0)
set_axes_equal(ax)
plt.show()