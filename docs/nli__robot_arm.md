---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


Recall the [dymanics](robot_arm.html) of the robot arm

$$
 \frac{d}{dt}\begin{pmatrix} \theta \\ \dot{\theta} \end{pmatrix} = 
 \begin{pmatrix}  \dot{\theta} \\ -a.\sin{\theta} -b.\dot{\theta} + c.\tau \end{pmatrix}
$$

It can be rearanged into the so-called `Control-Affine` form, i.e.

$$
\dot{X} = f(X) + g(X).U
$$

where

$$
f(X) = \begin{pmatrix}  \dot{\theta} \\ -a.\sin{\theta} -b.\dot{\theta}\end{pmatrix}
\quad \text{and} \quad
g(X) = \begin{pmatrix}  0 \\ c \end{pmatrix}
$$

In discrete time:

$$
y_{k+1} = f \left(y_k, y_{k-1} \dots y_{k-n+1}, u_{k-1} \dots u_{k-m+1}\right) +
 g \left(y_k, y_{k-1} \dots y_{k-n+1}, u_{k-1} \dots u_{k-m+1}\right). u_k
$$
