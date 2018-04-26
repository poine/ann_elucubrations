#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, numpy as np, matplotlib.pyplot as plt
import keras
import recurrentshop

import pdb


'''
training a plant model on a first order LTI using recurentshop readout
'''


# from recurrentshop/tests/test_recurrent_model.py
def test_readout():

    def sim(_u, a=-0.5, b=7., c=3.): # nb_step = len(_u) 
        _o = np.zeros((len(_u),1))
        #for i in range(1, len(_u)): _o[i] = np.tanh(a*(_o[i-1] + _u[i-1]) + b)
        for i in range(1, len(_u)): _o[i] = a*(_o[i-1] + _u[i-1]) + b
        return _o
    
    _dim = 1

    # Make recurrent model
    x = keras.layers.Input((_dim,), name='x')
    y_tm1 = keras.layers.Input((_dim,), name='y_tm1')
    h_tm1 = keras.layers.Input((_dim,), name='init_states')  # initial states
    x__p__y_tm1 = keras.layers.add([x, y_tm1], name='x__p__y_tm1')
    _d1 = keras.layers.Dense(_dim, name='d1')(x__p__y_tm1)
    _d2 = keras.layers.Dense(_dim, name='d2', use_bias=False)(h_tm1)
    h = keras.layers.add([_d1, _d2])
    #h = keras.layers.Activation('tanh')(h)

    rnn = recurrentshop.RecurrentModel(input=x, initial_states=h_tm1, output=h, final_states=h, readout_input=y_tm1, return_sequences=True)
    
    rnn.model.summary()
    print('d1: {}'.format(rnn.model.get_layer(name='d1').get_weights()))
    print('d2: {}'.format(rnn.model.get_layer(name='d2').get_weights()))

    # train it
    nb_steps = 7
    a = keras.layers.Input((nb_steps, _dim))
    if 1:
        b = rnn(a)
    else:
        foo__init_state = keras.layers.Input((_dim,), name='foo__init_states')  # initial states
        foo__init_readout = keras.layers.Input((_dim,), name='foo__init_readout')  # initial readout_input
        b = rnn(a, initial_state=foo__init_state, initial_readout=foo__init_readout)
    model = keras.models.Model(inputs=a, outputs=b)
    model.summary()
    
    nb_batch = 32
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    training_inputs = np.random.random((nb_batch, nb_steps, _dim))
    #training_outputs = np.random.random((nb_batch, nb_steps, _dim))
    training_outputs = np.array([sim(_u) for _u in  training_inputs])
    history = model.fit(training_inputs, training_outputs, epochs=1000, verbose=0, validation_split=0)


    # display training
    print(history.history.keys())
    plt.subplot(2,1,1)
    plt.plot(history.history['loss']); #plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.subplot(2,1,2)
    plt.plot(history.history['acc']); #plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    print('final loss {}'.format(history.history['loss'][-1]))
    plt.show()
    print('d1: {}'.format(rnn.model.get_layer(name='d1').get_weights()))
    print('d2: {}'.format(rnn.model.get_layer(name='d2').get_weights()))

    test_inputs = np.zeros((1, nb_steps, _dim))
    pred_out = model.predict(test_inputs)
    pdb.set_trace()


def test_readout2():
    '''let's forget initial state and bias for now'''
    def sim(_u, a=-0.5):
        _o = np.zeros((len(_u),1))
        for i in range(1, len(_u)): _o[i] = a*(_o[i-1] + _u[i-1])
        return _o

    _dim=1
    seq_len=10
    # Make recurrent model
    x = keras.layers.Input((_dim,), name='x')
    y_tm1 = keras.layers.Input((_dim,), name='y_tm1')
    x__p__y_tm1 = keras.layers.add([x, y_tm1], name='x__p__y_tm1')
    _d1 = keras.layers.Dense(_dim, name='d1', use_bias=False)(x__p__y_tm1)

    #rnn = recurrentshop.RecurrentModel(input=x, output=_d1, readout_input=y_tm1, final_states=_d1,  initial_states=_d1, return_sequences=True)
    rnn = recurrentshop.RecurrentModel(input=x, output=_d1, readout_input=y_tm1, return_sequences=True)
    rnn.model.summary()
    # train it
    a = keras.layers.Input((seq_len, _dim))
    b = rnn(a)
      
    # test
    test_input = np.ones(seq_len)
    test_output = sim(test_input)
    plt.plot(test_output)
    plt.show()



def test_readout3():
    ''' readout without recurrentshop....'''
    state_input = keras.layers.Input((1,), name ="x_k")   # x_k
    ctl_input = keras.layers.Input((1,), name ="u_k")     # u_k
    plant_input = keras.layers.concatenate([state_input, ctl_input])
    plant_l = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', input_shape=(2,), use_bias=False, name="plant")
    plant_output = plant_l(plant_input)

    model = keras.models.Model(inputs=ctl_input, outputs=plant_output)


    
def test2():
    ctl_i = keras.layers.Input((1,), name ='ctl_i')                          # u_k
    readout_i =  keras.layers.Input((1,), name='readout_i')                  # x_k
    plant_i = keras.layers.concatenate([ctl_i, readout_i], name='plant_i')   # u_k, x_k
    plant_l = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', input_shape=(2,), use_bias=False, name="plant_l")
    plant_o = plant_l(plant_i)
    rs_model = recurrentshop.RecurrentModel(input=ctl_i, output=plant_o, readout_input=readout_i)
    rs_model.model.summary()

    batch_size = 32
    x =  keras.layers.Input((batch_size, 1))
    y =  rs_model(x)
    #kr_model = keras.models.Model(x, y)
    
 
def test1():
    u =  keras.layers.Input((1,), name="control_input")
    readout_input =  keras.layers.Input((1,), name="readout_input")
    rnn_input =  keras.layers.add([u, readout_input])
    h_tm1 =  keras.layers.Input((1,), name="h_km1")

    cell = recurrentshop.SimpleRNNCell(1, name="dyn", use_bias=False)
    rnn_output, h = cell([rnn_input, h_tm1])
    nn = recurrentshop.RecurrentModel(input=u, initial_states=[h_tm1], output=rnn_output, final_states=[h], readout_input=readout_input)

    nn.model.summary()
    print nn.model.get_layer(name="dyn").get_weights()

    x =  keras.layers.Input((7,1))
    y = nn(x)
    model = keras.models.Model(x, y)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # train
    
    ann_input = np.zeros((100, 7,1))
    ann_output = np.zeros((100))
    model.fit(ann_input, ann_output, epochs=10, batch_size=32,  verbose=1, shuffle=True)
    
    
    # test
    _i = np.array([1, 1, 1, 0, 0, 0, -1]).reshape((1, 7, 1))
    _o = model.predict(_i)
    pdb.set_trace()
    plt.plot(_i.reshape(7), _o)
    plt.show()

    
def main():
    # fix random seed for reproducibility
    if 0:
        seed = 8; np.random.seed(seed)
    #test1()
    #test2()
    test_readout()  # 
    #test_readout2() # without initial state - doesn't work
    #test_readout3()  # without recurrentshop - looks like i can not do it...   

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
