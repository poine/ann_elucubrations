#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf, tflearn

import test_traj_pred as tpu
from test_traj_pred import Trajectory

import pdb

'''
here i will add predicition at different horizons
'''

class AnnAgent:
    def __init__(self, nps=2, nvs=2, horiz=10, ils=[]):
        self.nps, self.nvs = nps, nvs
        self.delay = max(nps, nvs)-1
        self.input_size = (nps+nvs-1)*2
        self.output_size = 2*horiz
        self.horiz = horiz
        input_tensor = tflearn.input_data(shape=[None, self.input_size], dtype=tf.float32, name='input_layer')
        l1 = input_tensor
        #for il in ils:
        #    l1 = tflearn.fully_connected(l1, il)
        out = tflearn.fully_connected(l1, self.output_size, activation='linear', name='output_layer')

        net = tflearn.regression(out, optimizer='sgd', loss='mean_square', learning_rate=0.001, metric=None)
        self.model = tflearn.DNN(net, tensorboard_dir='/tmp/traj_pred_tflearn_logs/',
                                 best_checkpoint_path='/tmp/traj_pred_best',
                                 checkpoint_path='/tmp/traj_pred_current')

    def _inp(self, traj, k):
        _ps = np.concatenate([traj.vel[k-i] for i in range(self.nvs)])
        _vs = np.concatenate([traj.pos[k-i]-traj.pos[k] for i in range(1, self.nps)]) if self.nps>1 else []
        _i = np.concatenate((_vs, _ps))
        return _i
        
    def prepare_training_set(self, ds):
        _input, _output = [], []
        for t in ds.trajectories:
            for k in range(self.delay, len(t.points)-self.horiz):
                _input.append(self._inp(t,k))
                _output.append((t.pos[k+1:k+self.horiz+1]-t.pos[k]).reshape(2*self.horiz))
        return np.array(_input), np.array(_output)
        
    def train(self, ds, epochs=100, run_id=None):
        _input, _output = self.prepare_training_set(ds)
        self.model.fit(_input, _output, n_epoch=epochs, batch_size=64, show_metric=True, validation_set=0.1, run_id=run_id)

    def save(self, filename='/tmp/traj_pred_final'):
        self.model.save(filename)

    def load(self, filename='/tmp/traj_pred_final'):
        self.model.load(filename)

    def predict(self, traj, k, horiz):
        _i = self._inp(traj, k).reshape((1,self.input_size))
        _j = 2*(int(horiz)-1)
        return self.model.predict(_i)[:,_j:_j+2] + traj.pos[k]

    def report(self):
        with self.model.session.as_default():
            ol_vars = tflearn.variables.get_layer_variables_by_name('output_layer')
            print('weights\n{}'.format(tflearn.variables.get_value(ol_vars[0])))
            print('bias\n{}'.format(tflearn.variables.get_value(ol_vars[1]))) 



def test(nps=2, nvs=2, horiz=5, epochs=10, ils=[]):
    d = tpu.DataSet(load='../data/bdx_20130914_25ft.pkl')
    a = AnnAgent(nps=nps, nvs=nvs, horiz=horiz, ils=ils)
    a.train(d, epochs=epochs, run_id='{}_{}_{}'.format(nps, nvs, ils))
    a.save()
    #a.load()
    a.report()
    tpu.test_predictions(a, d.trajectories, [horiz])
    plt.show()


def test_forced():
    nps, nvs, horiz, ils = 1, 1, 10, []
    a = AnnAgent(nps=nps, nvs=nvs, horiz=horiz, ils=ils)
    #d = tpu.DataSet(load='../data/bdx_20130914_25ft.pkl')
    #a.train(d, epochs=10, run_id='{}_{}_{}'.format(nps, nvs, ils))
    with a.model.session.as_default():
        print('forcing weights and biases')
        weights, biases = np.zeros((2, horiz*2)), np.zeros(horiz*2)
        for i in range(horiz):
            weights[0][2*i] = i+1
            weights[1][2*i+1] = i+1 
        ol_vars = tflearn.variables.get_layer_variables_by_name('output_layer')
        tflearn.variables.set_value(ol_vars[0], weights)
        tflearn.variables.set_value(ol_vars[1], biases)
    a.report()
    a.save('/tmp/traj_pred_tfl4_1_1_10_forced')
    

    
def main(train=True):
    #test(nps=9, nvs=9, horiz=10, epochs=100, ils=[])
    test_forced()

    
if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    main()
