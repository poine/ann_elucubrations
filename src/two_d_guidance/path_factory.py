#!/usr/bin/env python
import logging, sys, math, numpy as np

import path

import pdb

def pt_on_circle(c, r, th):
    return c + np.stack([r*np.cos(th), r*np.sin(th)], axis=-1)

def make_line_path(p0, p1, n_pt=10):
    _path = path.Path()
    _pts = np.stack([np.linspace(p0[i], p1[i], n_pt) for i in range(2)], axis=-1)
    disp = p1-p0
    yaws =  np.arctan2(disp[1], disp[0])*np.ones(len(_pts))
    _path.append_points(_pts, yaws)
    return _path

def make_circle_path(c, r, th0, th1, n_pt=10):
    _path = path.Path()
    ths = np.linspace(th0, th1, n_pt)
    pts = pt_on_circle(c, r, ths)
    _path.append_points(pts, ths+math.pi/2)
    return _path

def make_oval_path(c1, c2, r):
    circle1 = make_circle_path(c1, r, -math.pi, 0, n_pt=100)
    circle2 = make_circle_path(c2, r, 0, math.pi, n_pt=100)
    line1 = make_line_path(circle2.points[-1], circle1.points[0], n_pt=100)
    line2 = make_line_path(circle1.points[-1], circle2.points[0], n_pt=100)
    line1.append([circle1, line2, circle2])
    return line1


def make_fig_of_height_path(c, d, r):
    c1 = c + [0, d]
    circle1 = make_circle_path(c1, -r, -math.pi, 0, n_pt=100)
    c2 = c + [0, -d]
    circle2 = make_circle_path(c2, -r,  math.pi, 0, n_pt=100)
    line1 = make_line_path(circle1.points[-1], circle2.points[0], n_pt=100)
    line2 = make_line_path(circle2.points[-1], circle1.points[0], n_pt=100)
    circle1.append([line1, circle2, line2])
    
    return circle1


def main(args):

    # y_aligned line
    p0, p1 = np.array([0.45, 0.45]), np.array([0.45, 3.15])
    _path = make_line_path(p0, p1)
    _path.save('/home/poine/work/oscar.git/oscar/oscar_control/paths/line_01.npz')

    # x_aligned line
    p2 = np.array([1.95, 0.45])
    _path = make_line_path(p0, p2)
    _path.save('/home/poine/work/oscar.git/oscar/oscar_control/paths/line_02.npz')

    # counter-clockwise half circle
    c1 = np.array([1.2, 1.2])
    _path = make_circle_path(c1, r=0.75, th0=-math.pi, th1=0, n_pt=100)
    _path.save('/home/poine/work/oscar.git/oscar/oscar_control/paths/arc_01.npz')
    
    # clockwise half circle
    c2 = np.array([1.2, 2.4])
    _path = make_circle_path(c2, r=-0.75, th0=-math.pi, th1=0, n_pt=100)
    _path.save('/home/poine/work/oscar.git/oscar/oscar_control/paths/arc_02.npz')

    # oval
    _path = make_oval_path(c1, c2, r=0.75)
    _path.save('/home/poine/work/oscar.git/oscar/oscar_control/paths/oval_01.npz')

def test():
    # figure of height
    p3 = np.array([1.2, 2.])
    _path = make_fig_of_height_path(p3, d=0.75, r=0.75)
    _path.save('/home/poine/work/oscar.git/oscar/oscar_control/paths/foh_01.npz')
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #main(sys.argv)
    test()
