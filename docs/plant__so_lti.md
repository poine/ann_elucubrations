---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


# (Single Input) Second Order Linear Time Invariant (LTI) Plant 

The general form of a discrete time state space representation for a second order single input linear time invariant plant is as follow:

$$
\begin{pmatrix}x_1 \\ x_2 \end{pmatrix}_{k+1} = 
\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22}\end{pmatrix} \begin{pmatrix}x_1 \\ x_2 \end{pmatrix}_{k} +
\begin{pmatrix} b_{1} \\ b_{2} \end{pmatrix} \begin{pmatrix} u \end{pmatrix}_{k}
$$
 
 where $$\begin{pmatrix}x_1 \\ x_2 \end{pmatrix}_{k}$$ is the state vector, and $$ \begin{pmatrix} u \end{pmatrix}_{k} $$ is the input vector.
