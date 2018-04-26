#! /usr/bin/env python
# -*- coding: utf-8 -*-

#
# We are using a pair of pressure sensors to determine airspeed and angle of attack in a fixed wing aircraft 
#
# A_p is the pressure sensor measuring the AoA
# V_p is the pressure sensor measuring the airspeed
# Theta is the pitch angle coming from IMU, which is equal to real AoA in wind tunnel
# Airspeed is the value we measure in the wind tunnel during the experiments.
#

import logging, timeit, math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils as ut

def read_dataset(filename):
    if 1:
        unused, Vp, Ap, theta, airspeed = np.loadtxt(filename, skiprows=1, delimiter=',', unpack=True )
    else:
        with open(filename, "rb") as f:
            for i, line in enumerate(f):
                print('{} -> {}'.format(i,line))
                print line.split(',')
    return Vp, Ap, theta, airspeed



def remove_stalled_data(Vp, Ap, theta, airspeed, theta_max=ut.rad_of_deg(14), theta_min=ut.rad_of_deg(-14.)):
    ''' Throw away data where airfoil is stalled '''
    _l1 = len(theta)
    sel = ((theta <= theta_max) & (theta >= theta_min))
    _l2 = len(theta[sel])
    print('removed {} stalled points (>{} deg or <{} deg) ({} of {} remaining)'.format(_l1-_l2, ut.deg_of_rad(theta_max), ut.deg_of_rad(theta_min), _l2, _l1))
    return Vp[sel], Ap[sel], theta[sel], airspeed[sel]
    

def uniformize_data(Vp, Ap, theta, airspeed, _nbins = 100, max_samples = 15):
    ''' Uniformization
     We randomly select a maximum number of points in regions where we have too many of them '''
    H, xedges, yedges = np.histogram2d(Vp, Ap, bins=_nbins)
    H2 = np.empty((_nbins, _nbins), dtype=object)
    for i in range(_nbins):
        for j in range(_nbins):
            H2[i,j] = []

    H3 = np.zeros(H.shape)
    x0, y0 = xedges[0], yedges[0]
    dx, dy = xedges[-1] - xedges[0],  yedges[-1] - yedges[0]
    for i in range(len(Vp)):
        xidx = min(int((Vp[i]-x0)/dx*_nbins), _nbins-1)
        yidx = min(int((Ap[i]-y0)/dy*_nbins), _nbins-1)
        H2[xidx, yidx].append(i)
        H3[xidx, yidx] += 1
    print 'recreated histogram: done'
    print('array_equal {}'.format(np.array_equal(H, H3)))
    
    Vp2, Ap2, theta2, airspeed2 = [],[],[],[]
    def add_samples(_l):
        for _j in _l:
            Vp2.append(Vp[_j]); Ap2.append(Ap[_j]); theta2.append(theta[_j]); airspeed2.append(airspeed[_j])
    for i in range(_nbins):
        for j in range(_nbins):
            if len(H2[i,j]) > max_samples:
                add_samples(np.random.choice(H2[i,j], max_samples))
            else:
                add_samples(H2[i,j])
    
    print('cook data: uniformized samples, kept {} points'.format(len(Vp2)))
    return np.array(Vp2), np.array(Ap2), np.array(theta2), np.array(airspeed2) 


def cook_data(Vp, Ap, theta, airspeed, theta_max=ut.rad_of_deg(14), theta_min=ut.rad_of_deg(-14.)):
    ''' Preprocessing of training data '''
    Vp2, Ap2, theta2, airspeed2 = remove_stalled_data(Vp, Ap, theta, airspeed, theta_max, theta_min)
    Vp3, Ap3, theta3, airspeed3 = uniformize_data(Vp2, Ap2, theta2, airspeed2)
    return  Vp3, Ap3, theta3, airspeed3
 
   



def analyze_data(Vp, Ap):
    ''' Plot histograms of the dataset '''
    fig = plt.figure()
    ax = plt.subplot(1,3,1)
    plt.hist(Vp, bins=100)
    ut.decorate(ax, title='Vp')
    ax = plt.subplot(1,3,2)
    plt.hist(Ap, bins=100)
    ut.decorate(ax, title='Ap')

    H, xedges, yedges = np.histogram2d(Vp, Ap, bins=100)
    ax = plt.gcf().add_subplot(133, title='2D histogram',
                         aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    im.set_data(xcenters, ycenters, H)
    ax.images.append(im)
    ut.decorate(ax, xlab='Vp', ylab='Ap')

    # 3D surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    (xpos, ypos), zpos = np.meshgrid( xcenters, ycenters), H
    surf = ax.plot_surface(xpos, ypos, zpos, cmap=mpl.cm.coolwarm, linewidth=0, antialiased=True, rstride=1, cstride=1)
    fig.colorbar(surf, shrink=0.5, aspect=5)



def plot_sequential(Vp, Ap, theta, airspeed, pred_theta=None, pred_airspeed=None):
    plt.figure()
    ax = plt.subplot(2,2,1)
    plt.plot(Vp)
    ut.decorate(ax, title='Vp')

    ax = plt.subplot(2,2,2)
    plt.plot(Ap)
    ut.decorate(ax, title='Ap')

    ax = plt.subplot(2,2,3)
    plt.plot(ut.deg_of_rad(theta))
    if pred_theta is not None: plt.plot(ut.deg_of_rad(pred_theta))
    ut.decorate(ax, title='theta', ylab='deg')

    ax = plt.subplot(2,2,4)
    plt.plot(airspeed)
    if pred_airspeed is not None: plt.plot(pred_airspeed)
    ut.decorate(ax, title='airspeed', ylab='m/s')


def plot_pred_err(pred_out,  test_output):
    plt.figure()
    pred_err = (pred_out - test_output)
    mus, sigmas = np.mean(pred_err, axis=0), np.std(pred_err, axis=0)
    ax = plt.subplot(1,2,1)
    plt.hist(ut.deg_of_rad(pred_err[:,0]), bins=100)
    ut.decorate(ax, title='Theta err', xlab='deg', legend=['$\mu$ {:.3f} deg $\sigma$ {:.3f} deg'.format(ut.deg_of_rad(mus[0]), ut.deg_of_rad(sigmas[0]))])
    ax = plt.subplot(1,2,2)
    plt.hist(pred_err[:,1], bins=100)
    ut.decorate(ax, title='Airspeed err', xlab='m/s', legend=['$\mu$ {:.3f} m/s $\sigma$ {:.3f} m/s'.format(mus[1], sigmas[1])])
