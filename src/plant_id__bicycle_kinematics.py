#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, math, numpy as np, matplotlib.pyplot as plt, pickle, os
import keras, control
import bicycle_kinematics, utils as ut
import pdb
LOG = logging.getLogger('plant_id__bicycle_kinematics')



def analyse_dataset(time, X, U, exp_name):
    margins = (0.04, 0.07, 0.98, 0.93, 0.27, 0.2)
    figure = ut.prepare_fig(figsize=(20.48, 7.68), margins=margins)
    plots = [('$x$', 'm', X[:,0]),
             ('$y$', 'm', X[:,1]),
             ('$\psi$', 'deg', ut.deg_of_rad(X[:,2])),
             ('$v$', 'm/s', X[:,3]),
             ('$a$', 'm/s2', U[:,0]),
             ('$df$', 'deg/s', ut.deg_of_rad(U[:,1]))]
    
    for i, (_ti, _un, _d) in enumerate(plots):
        ax = plt.subplot(1,6,i+1)
        plt.hist(_d, bins=100)
        ut.decorate(ax, title=_ti, xlab=_un)
    ut.save_if('../docs/plots/plant_id__bycicle_kinematics__{}_training_set_histogram.png'.format(exp_name))


def make_uniform_training_set(plant, dt, force_remake=False, nsamples=int(50e3)):
    filename = '/tmp/bicycle_kinematics__uniform_training_traj.pkl'
    if force_remake or not os.path.isfile(filename):
        desc = 'uniform mip trajectory'
        Xks = np.vstack((np.random.uniform(-1., 1., nsamples), # x
                         np.random.uniform(-1., 1., nsamples), # y
                         np.random.uniform(-math.pi, math.pi, nsamples), # psi
                         np.random.uniform(-3., 3., nsamples)  # v
                        )).T
        Uks = np.vstack((np.random.uniform(-0.5, 0.5, nsamples), # a
                         np.random.uniform(-ut.rad_of_deg(45), ut.rad_of_deg(45), nsamples) # df
                        )).T
        Xkp1s = np.zeros((nsamples, 4))
        for k in range(nsamples):
            Xkp1s[k] = plant.disc_dyn(Xks[k], Uks[k], dt)
        with open(filename, "wb") as f:
            pickle.dump([Xks, Uks, Xkp1s, desc], f)
    else:
        with open(filename, "rb") as f:
          Xks, Uks, Xkp1s, desc = pickle.load(f)
    _input =  np.hstack([Xks, Uks])
    _output = Xkp1s
    time = dt*np.arange(len(_input))
    return time, Xks, Uks, desc, _input, _output

def ident_plant(_input, _output, expe_name, epochs=50, force_train=False, display_training_history=False):
    filename = "/tmp/plant_id__bicycle_kinematics.h5"
    if force_train or not os.path.isfile(filename):

        plant_i = keras.layers.Input((6,), name ="plant_i") # x1_k, x2_k, x3_k, x4_k, u1_k, u2_k
        if 0:
            plant_l = keras.layers.Dense(4, activation='linear', kernel_initializer='uniform', use_bias=False, name="plant")
            plant_o = plant_l(plant_i)
        else:
            plant_l1 = keras.layers.Dense(24, activation='relu', kernel_initializer='uniform', use_bias=True, name="plantl1")
            plant_l2 = keras.layers.Dense(24, activation='relu', kernel_initializer='uniform', use_bias=True, name="plantl2")
            plant_l3 = keras.layers.Dense(4, activation='linear', kernel_initializer='uniform', use_bias=False, name="plantl3")
            plant_o = plant_l3(plant_l2(plant_l1(plant_i)))
             
        plant_ann = keras.models.Model(inputs=plant_i, outputs=plant_o)
        plant_ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)
        history = plant_ann.fit(_input, _output, epochs=epochs, batch_size=32,  verbose=1, shuffle=True, validation_split=0.1, callbacks=[early_stopping])
        
        if display_training_history:
            margins = (0.04, 0.07, 0.98, 0.93, 0.27, 0.2)
            figure = ut.prepare_fig(figsize=(20.48, 7.68), margins=margins)
            ax = plt.subplot(1,2,1); ut.decorate(ax, title='loss'); plt.plot(history.history['loss'])
            ax = plt.subplot(1,2,2); ut.decorate(ax, title='accuracy'); plt.plot(history.history['acc'])
            ut.save_if('../docs/plots/plant_id__bicycle_kinematics__{}_training_history.png'.format(expe_name))
        plant_ann.save(filename)
    else:
        plant_ann = keras.models.load_model(filename)
    return plant_ann



def main(force_remake_training_set=False, display_training_set=True):
    dt = 0.01
    plant = bicycle_kinematics.Plant()

    exp_name = "uniform"
    time, X, U, desc, _input, _output = make_uniform_training_set(plant, dt, force_remake_training_set)
    if display_training_set: analyse_dataset(time, X, U, exp_name)
    plant_ann = ident_plant(_input, _output, exp_name, epochs=50, force_train=True, display_training_history=True)
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
