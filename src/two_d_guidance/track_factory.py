
import logging, sys, math, numpy as np
import matplotlib.pyplot as plt

from. import track, path, path_factory, utils

def make_circle_track(c, r1, r2, th0, dth, n_pt=10):
    r = (r1+r2)/2
    _track = track.Track()
    ths = np.linspace(th0, th0+np.sign(r)*dth, n_pt, endpoint=False)
    _track.left_border = utils.pt_on_circle(c, np.abs(r1), ths)
    _track.right_border =  utils.pt_on_circle(c, np.abs(r2), ths)
    _track.path = path_factory.make_circle_path(c, r, th0, dth, n_pt)
    return _track


def make_oval_track(c1, c2, r, w):
    _track = track.Track()
    _track.path = path_factory.make_oval_path(c1, c2, r)
    _track.left_border = np.array(_track.path.points)
    _track.right_border = np.array(_track.path.points)
    for i in range(len(_track.path.points)):
        v = np.array([np.cos(_track.path.headings[i]+np.pi/2), np.sin(_track.path.headings[i]+np.pi/2)])
        _track.left_border[i] += v*w/2
        _track.right_border[i] -= v*w/2
    return _track

def make_fig_of_height_track(r, w):
    _track = track.Track()
    _track.path = path_factory.make_fig_of_height_path2(r)
    _track.left_border = np.array(_track.path.points)
    _track.right_border = np.array(_track.path.points)
    for i in range(len(_track.path.points)):
        v = np.array([np.cos(_track.path.headings[i]+np.pi/2), np.sin(_track.path.headings[i]+np.pi/2)])
        _track.left_border[i] += v*w/2
        _track.right_border[i] -= v*w/2
    return _track


def view(_track):
    fig, ax = plt.gcf(), plt.subplot(1,1,1)
    path_factory.draw_path(fig, ax, _track.path)
    plt.plot(_track.left_border[:,0], _track.left_border[:,1], 'r', linewidth=0.5)
    plt.plot(_track.right_border[:,0], _track.right_border[:,1], 'g', linewidth=0.5)

    plt.fill(_track.right_border[:,0], _track.right_border[:,1], alpha=0.2, color='grey')
    plt.fill(_track.left_border[:,0], _track.left_border[:,1], color='white')

    xl_min, xl_max = _track.left_border[:,0].min(), _track.left_border[:,0].max()
    xr_min, xr_max = _track.right_border[:,0].min(), _track.right_border[:,0].max()
    x_min, x_max  = min(xl_min, xr_min), max(xl_max, xr_max)
    
    yl_min, yl_max = _track.left_border[:,1].min(), _track.left_border[:,1].max()
    yr_min, yr_max = _track.right_border[:,1].min(), _track.right_border[:,1].max()
    y_min, y_max  = min(yl_min, yr_min), max(yl_max, yr_max)
    dx, dy = x_max - x_min, y_max - y_min
    mx, my = 0.1*dx, 0.1*dy
    ax.set_xlim(x_min-mx, x_max+mx)
    ax.set_ylim(y_min-my, y_max+my)
    
    ax.set_aspect('equal'); plt.title('2D')
