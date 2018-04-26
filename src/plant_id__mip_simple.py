#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, numpy as np, matplotlib.pyplot as plt, pickle, os
import keras, control
import mip_simple, utils as ut
import pdb
LOG = logging.getLogger('plant_id__mip_simple')

'''
Plant ID on MIP simple
'''
class CtlPlaceFullPoles:
    def __init__(self, plant, dt):
        A, B = ut.num_jacobian([0, 0, 0, 0], [0], plant.dyn_cont)
        poles = [-33, -3.5+3.9j, -3.5-3.9j, -3.9]
        self.K = control.matlab.place(A, B, poles)
        print('K {}'.format(self.K))
        print('cl poles {}'.format(np.linalg.eig(A-np.dot(B, self.K))[0]))

    def get(self, X, i):
        dX = X - [self.x_sp[i], 0, 0, 0]
        return -np.dot(self.K, dX)

    
def make_controlled_training_set(plant, dt, force_remake=False, nsamples=int(10*1e3), max_nperiod=10, max_intensity=0.6):
    filename = '/tmp/mip_simple__training_traj.pkl'
    if force_remake or not os.path.isfile(filename):
        desc = 'controlled mip trajectory'
        ctl = CtlPlaceFullPoles(plant, dt)
        time, ctl.x_sp = ut.make_random_pulses(dt, nsamples, max_nperiod=max_nperiod, min_intensity=-max_intensity, max_intensity=max_intensity)
        X0 = [0, 0, 0, 0]
        X, U = plant.sim_with_input_fun(time, ctl, X0)
        ut.save_trajectory(time, X, U, desc, filename)
    else:
        time, X, U, desc = ut.load_trajectory(filename)
    _input = np.hstack([X[:-1], U[:-1]])
    _output = X[1:]
    return time, X, U, desc, _input, _output


def make_uniform_training_set(plant, dt, force_remake=False, nsamples=int(50e3)):
    filename = '/tmp/mip_simple__uniform_training_traj.pkl'
    if force_remake or not os.path.isfile(filename):
        desc = 'uniform mip trajectory'
        Xks = np.vstack((np.random.uniform(-1., 1., nsamples), # x
                         np.random.uniform(-ut.rad_of_deg(45), ut.rad_of_deg(45), nsamples), # theta
                         np.random.uniform(-3., 3., nsamples), # xdot
                         np.random.uniform(-ut.rad_of_deg(500), ut.rad_of_deg(500), nsamples), # theta dot
        )).T
        Uks = np.random.uniform(-0.5, 0.5, (nsamples,1))
        Xkp1s = np.zeros((nsamples, 4))
        for k in range(nsamples):
            Xkp1s[k] = plant.dyn_disc(Xks[k], 0, dt, Uks[k])
        with open(filename, "wb") as f:
            pickle.dump([Xks, Uks, Xkp1s, desc], f)
    else:
        with open(filename, "rb") as f:
          Xks, Uks, Xkp1s, desc = pickle.load(f)
    _input =  np.hstack([Xks, Uks])
    _output = Xkp1s
    time = dt*np.arange(len(_input))
    return time, Xks, Uks, desc, _input, _output

    

def analyse_dataset(time, X, U, exp_name):
    margins = (0.04, 0.07, 0.98, 0.93, 0.27, 0.2)
    figure = ut.prepare_fig(figsize=(20.48, 7.68), margins=margins)
    plots = [('$x$', 'm', X[:,0]),
             ('$\\theta$', 'deg', ut.deg_of_rad(X[:,1])),
             ('$\dot{x}$', 'm/s', X[:,2]),
             ('$\dot{\\theta}$', 'deg/s', ut.deg_of_rad(X[:,3])),
             ('$\\tau$', 'N', U[:,0])]
    
    for i, (_ti, _un, _d) in enumerate(plots):
        ax = plt.subplot(1,5,i+1)
        plt.hist(_d, bins=100)
        ut.decorate(ax, title=_ti, xlab=_un)
    ut.save_if('../docs/plots/plant_id__mip_simple__{}_training_set_histogram.png'.format(exp_name))



def ident_plant(_input, _output, expe_name, epochs=50, force_train=False, display_training_history=False):
    filename = "/tmp/plant_id__mip_simple.h5"
    if force_train or not os.path.isfile(filename):
        plant_i = keras.layers.Input((5,), name ="plant_i") # x1_k, x2_k, x3_k, x4_k, u_k
        if 1:
            plant_l = keras.layers.Dense(4, activation='linear', kernel_initializer='uniform', use_bias=False, name="plant")
            plant_o = plant_l(plant_i)
        else:
            plant_l1 = keras.layers.Dense(8, activation='relu', kernel_initializer='uniform', use_bias=True, name="plant1")
            plant_l2 = keras.layers.Dense(12, activation='relu', kernel_initializer='uniform', use_bias=True, name="plant2")
            plant_l3 = keras.layers.Dense(4, activation='linear', kernel_initializer='uniform', use_bias=True, name="plant3")
            plant_o = plant_l3(plant_l2(plant_l1(plant_i)))
        plant_ann = keras.models.Model(inputs=plant_i, outputs=plant_o)
        plant_ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
        history = plant_ann.fit(_input, _output, epochs=epochs, batch_size=32,  verbose=1, shuffle=True, validation_split=0.1, callbacks=[early_stopping])
        

        
        if display_training_history:
            margins = (0.04, 0.07, 0.98, 0.93, 0.27, 0.2)
            figure = ut.prepare_fig(figsize=(20.48, 7.68), margins=margins)
            ax = plt.subplot(1,2,1); ut.decorate(ax, title='loss'); plt.plot(history.history['loss'])
            ax = plt.subplot(1,2,2); ut.decorate(ax, title='accuracy'); plt.plot(history.history['acc'])
            ut.save_if('../docs/plots/plant_id__mip_simple__{}_training_history.png'.format(expe_name))
        plant_ann.save(filename)
    else:
        plant_ann = keras.models.load_model(filename)
    return plant_ann

    
def validate(plant, ann, dt, expe_name):
    ctl = CtlPlaceFullPoles(plant, dt)
    time = np.arange(0, 10., dt); ctl.x_sp = ut.step_vec(time, a0=-0.2, a1=0.2)
    X0 = [0., 0.01, 0, 0]
    X, U = plant.sim_with_input_fun(time, ctl, X0)

    Xm, Um = np.zeros((len(time), 4)), np.zeros((len(time), 1))
    Xm[0] = X0
    for k in range(1, len(time)):
        Um[k-1] = ctl.get(Xm[k-1], k-1)
        Xm[k] = ann.predict(np.array([[Xm[k-1,0], Xm[k-1,1], Xm[k-1,2], Xm[k-1,3], Um[k-1,0]]]))
    figure = mip_simple.plot_short(time, X, U)
    mip_simple.plot_short(time, Xm, Um, figure=figure)
    plt.subplot(3,1,1); plt.legend(['plant', 'ann'])
    plt.savefig('../docs/plots/plant_id__mip_simple__{}_fs.png'.format(expe_name))
    plt.show()


def main(force_remake_training_set=False, display_training_set=True):
    dt = 0.01
    plant = mip_simple.Plant()
    #exp_name = "controlled"
    #time, X, U, desc, _input, _output = make_controlled_training_set(plant, dt, force_remake=force_remake_training_set)
    exp_name = "uniform"
    time, X, U, desc, _input, _output = make_uniform_training_set(plant, dt)

    if display_training_set:
        mip_simple.plot_short(time, X, U, filename='../docs/plots/plant_id__mip_simple__{}_training_set.png'.format(exp_name))
        analyse_dataset(time, X, U, exp_name)
        plt.show()

    ann = ident_plant(_input, _output, exp_name, epochs=150, force_train=True, display_training_history=True)
    validate(plant, ann, dt, exp_name)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
