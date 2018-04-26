#! /usr/bin/env python
# -*- coding: utf-8 -*-


''' 
   this is a (firtst order LTI) plant experiment with RNM
   Now I can train it!!!!  the trick was to use stateless for training
   X_kp1 = a x_k + b u_k
'''


import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import keras, sklearn.neural_network
import utils as ut, pdb
# reading 
# https://github.com/keras-team/keras/blob/master/examples/lstm_stateful.py


LOG = logging.getLogger('test_keras_rnn')

#import test_rnn as trnn
class Plant:
    def __init__(self, tau=1., dt=0.01):
        self.tau, self.dt = tau, dt
        self.ad, self.bd = np.exp(-dt/tau), 1. - np.exp(-dt/tau)
        LOG.info('  Plant ad {} bd {}'.format(self.ad, self.bd))

    def cont_dyn(self, X, t, U):
        Xd =  -1./self.tau*(X-U)
        return Xd

    def disc_dyn(self, Xk, Uk):
        _unused, Xkp1 = scipy.integrate.odeint(self.cont_dyn, Xk, [0, self.dt], args=(Uk,))
        return Xkp1

    def disc_dyn2(self, Xk, Uk):
        Xkp1 = self.ad*Xk+self.bd*Uk
        return Xkp1

    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), 1)),  np.zeros((len(time), 1))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            X[i] = self.disc_dyn(X[i-1], U[i-1])
        U[-1] = U[-2]
        return X, U

class CtlNone:
    def __init__(self, yc=None):
        self.yc = yc

    def get(self, X, k):
        return self.yc[k]

def plot(time, X, U=None, Yc=None):
    ax = plt.subplot(2,1,1)
    plt.plot(time, X[:,0])
    if Yc is not None: plt.plot(time, Yc, 'k')
    ut.decorate(ax, title="$x_1$", ylab='time')
    if U is not None:
        ax = plt.subplot(3,1,3)
        plt.plot(time, U)
        ut.decorate(ax, title="$u$", ylab='time')

        
def make_dataset():
    dt, nsamples, max_nperiod = 0.01, 1000, 10
    time_tr, yc = ut.make_random_pulses(dt, nsamples, max_nperiod=max_nperiod,  min_intensity=-1, max_intensity=1.)
    plant = Plant(dt=dt)
    ctl = CtlNone(yc)
    X0 = [0.]
    Xtr, Utr = plant.sim(time_tr, X0, ctl.get)

    time_te =  np.arange(0., 15.05, plant.dt)
    ctl.yc = 2*ut.step_vec(time_te, dt=8)
    X0 = [0.5]
    Xte, Ute = plant.sim(time_te, X0, ctl.get)

    return time_tr, Xtr, Utr, time_te, Xte, Ute


def rework_training_set(Xtr, Utr, batch_len=2):
    n_samples = len(Xtr)
    
    #pdb.set_trace()


def test_functional_stateless():
    time_tr, Xtr, Utr, time_te, Xte, Ute = make_dataset()

    # train a stateless model
    input_tensor = keras.layers.Input((1,1))
    initial_state_tensor = keras.layers.Input((1,1))
    rnn_layer = keras.layers.SimpleRNN(units=1,
                                       input_shape=(1, 1),
                                       batch_size=1,
                                       stateful=False,
                                       use_bias=False,
                                       return_state=False,
                                       unroll=False,
                                       activation='linear',
                                       name="rnn_layer")
    
    output_tensor = rnn_layer(input_tensor, initial_state=initial_state_tensor)
    model = keras.models.Model([input_tensor, initial_state_tensor], output_tensor)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    
    _input, _output = [Utr[:-1].reshape((len(Utr)-1, 1, 1)), Xtr[:-1].reshape((len(Xtr)-1, 1, 1))], Xtr[1:]
    model.fit(_input, _output, epochs=10,  verbose=1, batch_size=1)
    print('rnn weights: {}'.format(model.get_layer(name="rnn_layer").get_weights()))

    # Test a stateful model with identical coefficients (just because it is simpler)
    rnn_layer2 = keras.layers.SimpleRNN(units=1,
                                        input_shape=(1, 1),
                                        batch_size=1,
                                        stateful=True,
                                        use_bias=False,
                                        return_state=False,
                                        unroll=False,
                                        activation='linear',
                                        name="rnn_layer2")
    model2 = keras.models.Sequential()
    model2.add(rnn_layer2)
    # copy weights from previously trained model
    model2.get_layer(name="rnn_layer2").set_weights(model.get_layer(name="rnn_layer").get_weights())
    model2.summary()

    _input_test = Ute.reshape((len(Ute), 1, 1))
    model2.get_layer(name="rnn_layer2").reset_states(states=np.array([Xte[0]]))
    Xte2 = model2.predict(_input_test, batch_size=1)
    plot(time_te, Xte, Ute)
    plot(time_te, Xte2, Ute)
    plt.subplot(2,1,1); plt.legend(['plant','ann'])
    plt.show()
    

def test_functional_statefull():
    input_tensor = keras.layers.Input((1,1), batch_shape=(32,1,1))
    state_tensor = keras.layers.Input((1,1))
    rnn_layer = keras.layers.SimpleRNN(units=1,
                                       input_shape=(1, 1),
                                       batch_size=1,
                                       stateful=True,
                                       use_bias=False,
                                       return_state=False,
                                       unroll=False,
                                       activation='linear',
                                       name="rnn_layer")
    output_tensor = rnn_layer(input_tensor, initial_state=state_tensor)

def test_control():
    ctl_ref_input_tensor = keras.layers.Input((1,1), batch_shape=(1,1,1),   name='ref_input')   # the reference input to control
    ctl_state_input_tensor = keras.layers.Input((1,1), batch_shape=(1,1,1), name='state_input') # the state input to control

    ctl_state_input_layer = keras.layers.Lambda(lambda x: x + 0., output_shape=lambda s: s)(ctl_state_input_tensor)
    
    ctl_input_tensor = keras.layers.Concatenate()([ctl_ref_input_tensor, ctl_state_input_tensor])

    ctl_layer = keras.layers.Dense(units=1, input_shape=((1, 2)), use_bias=False, trainable=False)

    ctl_output_tensor = ctl_layer(ctl_input_tensor)

    plant_layer = keras.layers.SimpleRNN(units=1,
                                         input_shape=(1, 1),
                                         batch_size=1,
                                         stateful=True,
                                         use_bias=False,
                                         return_state=True,
                                         unroll=False,
                                         activation='linear')

    plant_output_k_tensor, plant_output_kp1_tensor = plant_layer(ctl_output_tensor)

    model = keras.models.Model(ctl_ref_input_tensor, plant_output_kp1_tensor)
    pdb.set_trace()
    
    
def main():


    #test_functional_stateless()
    test_functional_statefull()
    #test_control()
    return
 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
