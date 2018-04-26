---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


First Order Linear Time Invariant (LTI) Plant Identification

The continuous-time model of a first order LTI system is given by:

$$
 \dot{x} = a_c x + b_c u = -\frac{1}{\tau} (x +gu) 
$$

which is discretized as

$$
 x_{k+1} = a_d x_k + b_d u_k
$$


FeedForward [sklearn code](https://github.com/poine/ann_elucubrations/blob/master/src/fo_lti_id_plant_feedforward_sklearn.py) 

FeedForward [keras code](https://github.com/poine/ann_elucubrations/blob/master/src/fo_lti_id_plant_feedforward_keras.py)

RNN: [keras code](https://github.com/poine/ann_elucubrations/blob/master/src/plant_id__fo__keras_rnn.py)
