import torch
import numpy as np
import matplotlib.pyplot as plt
from robot.groups import PlanarMobileRobot

def plot_2d_wall(ax, w, grid=False, locater=1, alpha=1,
                plot_axis=True, wall_color='black', axis_color='black',
                axis_width=3, linewidth=5, **kwargs):
    group = PlanarMobileRobot()
    w = w.reshape(1, 3)
    walls1, walls2, walls3, walls4 = group.get_wall(w)
    if grid:
        ax.grid(zorder=0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(locater))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(locater * 0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(locater))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(locater * 0.1))
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)
    # ax.axis('equal')
    
    ax.plot(walls1[0, 0], walls1[0, 1], color=wall_color, linewidth=linewidth, alpha=alpha)
    ax.plot(walls2[0, 0], walls2[0, 1], color=wall_color, linewidth=linewidth, alpha=alpha)
    ax.plot(walls3[0, 0], walls3[0, 1], color=wall_color, linewidth=linewidth, alpha=alpha)
    ax.plot(walls4[0, 0], walls4[0, 1], color=wall_color, linewidth=linewidth, alpha=alpha)

    if plot_axis:
        Rot_mat = group.rot_z(w.reshape(1, 3)[:, -1]).cpu()  # (n, 2, 2)
        wall_axis = torch.tensor([31.0, 0]).reshape(1, 2, 1)
        wall_axis = (Rot_mat@wall_axis).reshape(2, 1)
        wall_axis = torch.cat([torch.zeros(2, 1), wall_axis], dim=1)
        # wall_axis = torch.cat([-wall_axis, wall_axis], dim=1)
        # ax.plot(wall_axis[0], wall_axis[1], '--', color=axis_color, linewidth=axis_width,
        #         alpha=alpha)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

def plot_init_point(ax, w, color='black', marker='D', alpha=1, size=5):
    w = w.reshape(3)
    ax.scatter(w[0], w[1], marker=marker, s=size, zorder=10, color=color, alpha=alpha, linewidth=0)

def plot_traj(ax, traj, color='tab:orange', alpha=1, linewidth=2):
    traj = traj.reshape(-1, 2)
    ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=linewidth)
    
def scatter_traj(ax, traj, color='tab:orange', alpha=1, size=10):
    traj = traj.reshape(-1, 2)
    ax.scatter(traj[:, 0], traj[:, 1], color=color, alpha=alpha, s=size)