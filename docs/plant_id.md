---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Plant identification

In control systems, a plant is usually modeled by an ordinary differential equation

$$
\dot{X} = f_c(X, U)
$$

where $$X \quad C^1:\mathbb{R} -> \mathbb{R}^n$$ is the *state* of the system and $$U \quad C^0:\mathbb{R} -> \mathbb{R}^m$$ is its *input*


When working with digital systems, it is common practice to discretize the above continuous-time model by considering the input $$U$$ to be constant on the time interval $$[t, t+dt]$$ and use a difference equation like

$$
X_{k+1} = f_d(X_k, U_k)
$$

As a universal function approximator, an Artificial Neural Networks (ANN) can be trained to approximate a plant's dynamics.


### First Order Linear Time Invariant (LTI) System

The continuous-time model of a first order LTI system is given by:

$$
 \dot{x} = a_c x + b_c u = -\frac{1}{\tau} (x +gu) 
$$

which is discretized as

$$

 x_{k+1} = a_d x_k + b_d u_k

$$


#### sklearn example
[](code)


#### keras example
[](code)

