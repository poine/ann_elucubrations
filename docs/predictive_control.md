---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Predictive Control

Model Predictive Control is an advanced method that uses an approximate model of a plant to allow trajectory tracking.
At each time step, an optimal control problem is solved on a finite horizon, and the first step of the solution is applied.


Given a plant such as:

$$

X_{k+1} = f(X_k,U_k)

$$

and an approximate model of the plant

$$

X_{k+1} = \tilde{f}(X_k,U_k)

$$

Given a desired trajectory $$ X^{r}_k$$, a solution to the following optimal control problem

  * minimize

$$

(X_{k0+N}-X^r_{k0+N})^T . Q_N.(X_{k0+N}-X^r_{k0+N}) +\Sigma_{k=k0}^{k0+N-1}

$$

  * subject to
  
$$
\begin{cases}
a \\ b
\end{cases}
$$
