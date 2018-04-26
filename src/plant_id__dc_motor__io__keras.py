#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, numpy as np, matplotlib.pyplot as plt, pickle, os
import keras, control
import dc_motor, utils as ut
import pdb
LOG = logging.getLogger('plant_id__mip_simple')

'''
Plant ID on dc motor, IO, Keras
'''

def ident_plant(plant, dt, force_train=False, epochs=50, force_dataset=False):
    filename = "/tmp/plant_id__dc_motor__keras__io.h5"
    if force_train or not os.path.isfile(filename):
         plant_i = keras.layers.Input((6,), name ="plant_i") # th_k, th_km1, th_km2, u_k, u_km1, u_km2
         plant_l = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', use_bias=False, name="plant")
         plant_o = plant_l(plant_i)
         plant_ann = keras.models.Model(inputs=plant_i, outputs=plant_o)
         plant_ann.compile(loss='mean_squared_error', optimizer='adam')

         time, X, U, desc  = dc_motor.make_or_load_training_set(plant, force_dataset, filename = '/tmp/dc_motor_training_traj.pkl', nsamples=int(100*1e3))
         delay = 3
         _len = len(time) - delay
         _input = np.vstack([X[2:-1,0], X[1:-2,0], X[:-3,0], U[2:-1,0], U[1:-2,0], U[:-3,0]]).T
         _output = X[3:,0].reshape((_len, 1))
         pdb.set_trace()
         early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
         history = plant_ann.fit(_input, _output, epochs=epochs, batch_size=256,  verbose=1, shuffle=True,
                                 validation_split=0.2, callbacks=[early_stopping])
    
         print plant_l.get_weights()
         
def main():
    dt = 0.01
    plant = dc_motor.Plant()

    print plant.Ac, plant.Bc
    A, B = ut.num_jacobian([0, 0, 0], [0, 0], plant.cont_dyn)
    print A, B
    
    ann = ident_plant(plant, dt, force_train=False, epochs=100, force_dataset=True)
    #validate(plant, ann, dt)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
