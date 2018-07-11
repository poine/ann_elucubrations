#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, pickle, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf, tflearn

import traj_pred_utils as tpu
from traj_pred_utils import Trajectory

import pdb

'''
here i will add predicition at different horizons
'''

class Param:
    def __init__(self, **kwargs):
        self.nps, self.nvs = kwargs['nps'], kwargs['nvs']
        self.ils, self.horiz = kwargs['ils'], kwargs['horiz']
        self.comment = kwargs.get('comment', '')

class AnnAgent:
    def __init__(self, **kwargs):#nps, nvs, ils, horiz):
        if kwargs['load'] is not None:
            with open(kwargs['load']+'.pkl', 'rb') as f:
                self.p = pickle.load(f)
        else:
            self.p = Param(**kwargs)

        self.delay = max(self.p.nps, self.p.nvs)-1
        self.input_size = (self.p.nps+self.p.nvs-1)*2
        self.output_size = 2*self.p.horiz

        input_tensor = tflearn.input_data(shape=[None, self.input_size], dtype=tf.float32, name='input_layer')
        l1 = input_tensor
        for i,il in enumerate(self.p.ils):
            l1 = tflearn.fully_connected(l1, il, activation='relu', name='inner_layer_{}'.format(i))
        out = tflearn.fully_connected(l1, self.output_size, activation='linear', name='output_layer')

        net = tflearn.regression(out, optimizer='sgd', loss='mean_square', learning_rate=0.001, metric=None)
        self.model = tflearn.DNN(net, tensorboard_dir='/tmp/traj_pred_tflearn_logs/',
                                 best_checkpoint_path='/tmp/traj_pred_best',
                                 checkpoint_path='/tmp/traj_pred_current')
        if kwargs['load'] is not None:
            self.model.load(kwargs['load'])

    def _inp(self, traj, k):
        ## FIXME _ps and _vs mixed :(
        _ps = np.concatenate([traj.vel[k-i] for i in range(self.p.nvs)])
        _vs = np.concatenate([traj.pos[k-i]-traj.pos[k] for i in range(1, self.p.nps)]) if self.p.nps>1 else []
        _i = np.concatenate((_vs, _ps))
        return _i
        
    def prepare_training_set(self, ds):
        _input, _output = [], []
        for t in ds.trajectories:
            for k in range(self.delay, len(t.time)-self.p.horiz):
                _input.append(self._inp(t,k))
                _output.append((t.pos_true[k+1:k+self.p.horiz+1]-t.pos_true[k]).reshape(2*self.p.horiz))
        return np.array(_input), np.array(_output)
        
    def train(self, ds, epochs=100, run_id=None):
        _input, _output = self.prepare_training_set(ds)
        self.model.fit(_input, _output, n_epoch=epochs, batch_size=64, show_metric=True, validation_set=0.1, run_id=run_id)

    def save(self, filename):
        print('saving agent to {}'.format(filename))
        self.model.save(filename)
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(self.p, f)

 #   def load(self, filename):
 #       print('loading agent from {}'.format(filename))
 #       self.model.load(filename)

    def predict(self, traj, k, horiz):
        _i = self._inp(traj, k).reshape((1,self.input_size))
        _j = 2*(int(horiz)-1)
        return self.model.predict(_i)[:,_j:_j+2] + traj.pos[k]

    def predict2(self, traj, k):
        _i = self._inp(traj, k).reshape((1,self.input_size))
        _p = self.model.predict(_i)[0]
        return _p.reshape((self.p.horiz,2)) + traj.pos[k]

    def report(self, verb=False):
        report  = 'params: nps {} nvs {} ils {} horiz {}\n'.format(self.p.nps, self.p.nvs, self.p.ils, self.p.horiz)
        report += 'comment: {}'.format(self.p.comment)
        if verb:
            with self.model.session.as_default():
                ol_vars = tflearn.variables.get_layer_variables_by_name('output_layer')
                report += '\nweights\n{}'.format(tflearn.variables.get_value(ol_vars[0]))
                report += 'bias\n{}'.format(tflearn.variables.get_value(ol_vars[1]))
        return report


def main(**kwargs):
    if 'dataset' in kwargs:
        d = tpu.DataSet(load=kwargs['dataset'])
    kwargs['nps'], kwargs['nvs'], kwargs['horiz'], kwargs['ils'] = 9, 9, 20, [60, 60]
    a = AnnAgent(**kwargs)
    if int(kwargs['epochs']) != 0: a.train(d, epochs=int(kwargs['epochs']), run_id=kwargs.get('run_id', None))
    if kwargs['save'] is not None: a.save(kwargs['save'])
    #if 'load' is not None: a.load(kwargs['load'])
    print(a.report())
    #tpu.test_predictions(a, d.trajectories, [horiz])
    #plt.show()


def save_forced():
    nps, nvs, ils, horiz = 1, 1, [], 20
    a = AnnAgent(nps=nps, nvs=nvs, ils=ils, horiz=horiz, comment='zero order forced weights')
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
    print(a.report())
    a.save('../data/traj_pred/agent_0/a')
    

    
def test(args):
    #test(nps=9, nvs=9, horiz=10, epochs=100, ils=[], save='../data/traj_pred_tfl4_9_9_[]_10/agent')
    #test(nps=15, nvs=15, horiz=20, epochs=100, ils=[], save='../data/traj_pred_tfl4_15_15_[]_20/agent')

    #test(nps=15, nvs=15, ils=[60, 60], horiz=20, epochs=200, save='../data/traj_pred_tfl4_15_15__60_60__20/agent')
    #test(nps=15, nvs=15, ils=[60, 60], horiz=20, load='../data/traj_pred_tfl4_15_15__60_60__20/agent')
    
    #test(nps=15, nvs=15, ils=[60, 60], horiz=20, dataset=args['dataset'], epochs=int(args['epochs']), save=args['save'],
    #     load=args['load'])
    #test(nps=15, nvs=15, ils=[60, 60], horiz=20, load='/tmp/foo/agent')

    save_forced()

    
if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help='filename of the dataset', default='../data/bdx_20130914_25ft.pkl')
    parser.add_argument('--epochs',  help='nb of epoch to train the agent', default=0)
    parser.add_argument('--save',    help='filename to save the agent', default=None)
    parser.add_argument('--comment', help=' comment to store along the agent', default='')
    parser.add_argument('--load',    help='filename from which to load the agent', default=None)
    
    args = vars(parser.parse_args())
    print(args)
    main(**args)
    #test()
