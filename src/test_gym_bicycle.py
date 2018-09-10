#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, time, logging, yaml, numpy as np, matplotlib, matplotlib.pyplot as plt
import gym, gym_foo, utils as ut, two_d_guidance as tdg, bicycle_dynamics as bcd
LOG = logging.getLogger('test_gym_bicycle')
import pdb


def make_speedo_picture():
    w, h, my_dpi = 800, 800, 96
    plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    ax = plt.gca()
    ax.set_xlim((-1.2, 1.2)); ax.set_ylim((-1.2, 1.2))
    ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(bottom=0., top=1., left=0., right=1.)
    #disk1 = plt.Wedge((0, 0), 0.95, 0, 350, color='k', lw=8, fill=False)
    disk1 = plt.Circle((0, 0), 0.8, color='k', lw=8, fill=False)
    ax.add_artist(disk1)
    for v in np.arange(0, 3, 0.5):
        alpha = np.pi - v; ca, sa = np.cos(alpha), np.sin(alpha)
        ax.text(ca, sa, '{}'.format(v), ha='center', va='center', fontsize=60)
        x1, y1 = 0.7*ca, 0.7*sa
        x2, y2 = 0.8*ca, 0.8*sa
        ax.plot([x1, x2], [y1, y2], lw=8, color='k')

    
    plt.savefig('gym_foo/envs/assets/speedometer.png', dpi=my_dpi, transparent=True)
    plt.show()
    
def make_path(which='ellipse_03', _dir='/home/poine/work/ann_elucubrations/data/paths/'):
    paths = {
        # line
        'line_01':    lambda: tdg.make_line_path([0, 0], [1., 0], 100),
        # ccw 1m circle
        'circle_01':  lambda: tdg.make_circle_path([0, 0], 1., 0, 2*np.pi, 360),
        # cw 1m circle
        'circle_02':  lambda: tdg.make_circle_path([0, 0], -1., 0, 2*np.pi, 360),
        # ccw 0.5m circle
        'circle_03':  lambda: tdg.make_circle_path([0, 0], 0.5, 0, 2*np.pi, 360),
        # cw 0.5m circle
        'circle_04':  lambda: tdg.make_circle_path([0, 0], -0.5, 0, 2*np.pi, 360),
        # ccw 5m circle
        'circle_10':  lambda: tdg.make_circle_path([0, 5], 5., -0.5*np.pi, 2*np.pi, 360),
        # ellipses 
        'ellipse_01': lambda: tdg.make_ellipse_path([-1., 0], [1., 0], 2.2),
        'ellipse_02': lambda: tdg.make_ellipse_path([-1., 0], [1., 0], 2.2, cw=True),
        'ellipse_03': lambda: tdg.make_ellipse_path([-1.5, 0], [1.5, 0], 3.5 ),
        'ellipse_04': lambda: tdg.make_ellipse_path([-1.5, 0], [1.5, 0], 3.5, cw=True ),
        # ovals 
        'oval_01':    lambda: tdg.make_oval_path([0.3, 0], [-0.3, 0], 0.3),
        'oval_02':    lambda: tdg.make_oval_path([0.3, 0], [-0.3, 0], -0.3),
        'oval_03':    lambda: tdg.make_oval_path([0.5, 0], [-0.5, 0], 0.5),
        'oval_04':    lambda: tdg.make_oval_path([0.5, 0], [-0.5, 0], -0.5),
        'oval_05':    lambda: tdg.make_oval_path([0.75, 0], [-0.75, 0], 0.75),
        'oval_06':    lambda: tdg.make_oval_path([0.75, 0], [-0.75, 0], -0.75),
        'oval_07':    lambda: tdg.make_oval_path([1., 0], [-1, 0], 1.),
        'oval_08':    lambda: tdg.make_oval_path([1., 0], [-1, 0], -1.),
        # 45deg ccw 1m radius oval 
        'oval_09':    lambda: tdg.make_oval_path([1., 1.], [-1, -1], 1.),
        # 45deg cw 1m radius oval 
        'oval_10':    lambda: tdg.make_oval_path([1., 1.], [-1, -1], -1.),
        # x aligned ccw 0.5m radius oval 
        'oval_07':    lambda: tdg.make_oval_path([0.5, 0], [-0.5, 0], 0.5),
        # figure of height
        'fig_of_height_01': lambda: tdg.make_fig_of_height_path2(0.3),
        'fig_of_height_02': lambda: tdg.make_fig_of_height_path2(0.5),
        'fig_of_height_03': lambda: tdg.make_fig_of_height_path2(0.75),
        'fig_of_height_04': lambda: tdg.make_fig_of_height_path2(1.),
        'fig_of_height_05': lambda: tdg.make_fig_of_height_path2(1.25),
        'fig_of_height_10': lambda: tdg.make_fig_of_height_path2(10.),
        # clover 01
        'clover_01': lambda: tdg.make_clover_path(r1=.5, r2=0.25),
        # clover 02
        'clover_02': lambda: tdg.make_clover_path(r1=.25, r2=0.5),
        # elliptical figure of height
        'el_fig_of_height_01': lambda: tdg.make_el_fig_of_height_path(d1=1, d2=0.05),
        # elliptical figure of height
        'el_fig_of_height_02': lambda: tdg.make_el_fig_of_height_path(d1=1, d2=0.25),
        # elliptical figure of height
        'el_fig_of_height_10': lambda: tdg.make_el_fig_of_height_path(d1=12, d2=0.6),
        # inscribed circles
        'inscribed_circles_01': lambda: tdg.make_inscribed_circles_path(r1=1, r2=2),
        # inscribed circles
        'inscribed_circles_02': lambda: tdg.make_inscribed_circles_path(r1=0.4, r2=0.75),
        # spirale
        'spirale_01': lambda: tdg.make_spirale_path([0, 0], 0.5, 0, np.pi, 0.4, 180),
    }
    if which is None:
        for which in paths.keys():
            fname = os.path.join(_dir, which+'.npz')
            p = paths[which]()
            p.save(fname)
    else:
        fname = os.path.join(_dir, which+'.npz')
        p = paths[which]()
        p.save(fname)
    return p, fname
    
        
   
def view_path(_p):#fname):
    #_p = tdg.path.Path(load=fname)
    fig, ax = plt.gcf(), plt.subplot(2,1,1)
    # 2D
    #plt.plot(_p.points[:,0], _p.points[:,1], '.')
    points = _p.points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(_p.points))
    lc = matplotlib.collections.LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(np.arange(len(_p.points)))
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    if 1:
        cm = plt.get_cmap('jet') 
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(_p.points))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        step = 10
        scale = _p.dists[step]
        for k in range(0, len(_p.points), step):
            plt.arrow(_p.points[k,0], _p.points[k,1], scale*np.cos(_p.headings[k]), scale*np.sin(_p.headings[k]),
                      width=0.01, color=scalarMap.to_rgba(k))

    ax.set_xlim(_p.points[:,0].min(), _p.points[:,0].max())
    ax.set_ylim(_p.points[:,1].min(), _p.points[:,1].max())
    ax.set_aspect('equal'); plt.title('2D')
    # Points
    plt.subplot(4, 2, 5)
    plt.plot(_p.points[:,0], '.')
    plt.plot(_p.points[:,1], '.')
    plt.title('points')
    # headings
    plt.subplot(4, 2, 6)
    plt.plot(_p.headings, '.')
    plt.title('headings')
    # distances
    plt.subplot(4, 2, 7)
    plt.plot(_p.dists, '.')
    plt.title('distances')
    # curvature
    plt.subplot(4, 2, 8)
    plt.plot(_p.curvatures, '.')
    plt.title('curvatures')
    plt.show()
    

def test_sim_with_pp(_plot, _render, cfg):
    env = gym.make('bicycle-v0')
    try:
        env.load_config(cfg['env'])
    except AttributeError:
        pass
    env.seed(123)
    env.reset()#[-0.39811196, -1.77139936,  0.76838096,0,0,0]   )
    # for oval
    #ctl =  tdg.PurePursuitVelController(env.path_filename, env.bicycle.P, look_ahead=0.5, v_sp=1.7)
    ctl =  tdg.PurePursuitVelController(env.path_filename, env.bicycle.P, look_ahead=env.carrot_dists[0], v_sp=1.2)
    _time = np.arange(0., 30, env.dt)
    for i in range(1, len(_time)):
        U = ctl.compute([env.X[bcd.s_x], env.X[bcd.s_y]], env.X[bcd.s_psi], env.X[bcd.s_vx])
        env.step(U)
        #print(env.X)
        env.render()
    time.sleep(2)

def test_blaaa():
    env = gym.make('bicycle-v0')
    env.seed(123)
    env.reset([0, 0, 0, 0, 0, 0])
    _p = env.path
    plt.plot(_p.points[:,0], _p.points[:,1], '.')
    plt.gca().set_aspect('equal')
    #pdb.set_trace()
    plt.show()
    
def test_yaml(fname):
    with open(fname, 'r') as stream:
        cfg = yaml.load(stream)
    #print('cfg {}'.format(cfg))
    #print(cfg['actor']['noise-sigma'])
    return cfg
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300, suppress=True)
    p, fname = make_path()
    #p = tdg.path.Path(load='../data/paths/track_ethz_dual_01.npz')
    view_path(p)#fname)
    #tdg.check_curvature(p)
    #plt.show()
    #make_speedo_picture()
    cfg = test_yaml('ddpg_run_bicycle_01.yaml')
    test_sim_with_pp(_plot='plot' in sys.argv, _render= 'render' in sys.argv, cfg=cfg)
    #test_blaaa()
