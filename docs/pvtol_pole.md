---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


# Planar VTOL with Pole

The *PVTOL with Pole* is a planar simplification of the problem of balancing a pole on top of a quadrotor, as demonstrated by Raffaello d'Andrea in his [TED conference in 2013](https://youtu.be/w2itwFJCgFQ?t=184).


<figure>
	<img src="images/ted_pvtol_pole.jpg" alt="The Pulpit Rock" width="252">
	<figcaption>Fig1. -  Raffaello d'Andrea TED conference.</figcaption>
</figure>

The following figure describes the variables and parameters of the model:
<figure>
	<img src="drawings/pvtol_pole.svg.png" alt="The Pulpit Rock" width="262">
	<figcaption>Fig2. - PVTOL with pole.</figcaption>
</figure>

We will start by deriving a dynamic model for the PVTOL with pole system.

### Kinematic

Assumingt that the pole is constrained to pivot at the center of the PVTOL, the position of the center of mass of the pole is then given by:

$$
\begin{cases}
x_p = x - l \sin{\phi} \\
z_p = z + l \cos{\phi}
\end{cases}
$$

Computing the time derivative of the above equation, we get the velocity of the pole's center of mass as:

$$
\begin{cases}
\dot{x}_p = \dot{x} - l \dot{\phi} \cos{\phi} \\
\dot{z}_p = \dot{z} - l \dot{\phi} \sin{\phi}
\end{cases}
$$

Computing yet another time derivative, the acceleration of the pole's center of mass is obtained as:

$$
\begin{cases}
\ddot{x}_p = \ddot{x} - l \left(\ddot{\phi}\cos{\phi} - \dot{\phi}^2 \sin{\phi}\right) \\
\ddot{z}_p = \ddot{z} - l \left(\ddot{\phi}\sin{\phi} + \dot{\phi}^2 \cos{\phi}\right)
\end{cases}
$$

Noting $$v^2=\dot{x}^2+\dot{z}^2$$ and $$v_p^2=\dot{x}_p^2+\dot{z}_p^2$$, the relationship between the pvtol and pole centers of mass velocities is expressed as:

$$
v_p^2 = \dot{x}^2 + \dot{z}^2 + l^2\dot{\phi}^2 -2l\dot{\phi}\left( \dot{x}\cos{\phi}+\dot{z}\sin{\phi}\right)
$$

$$
v_p^2 = v^2 + l^2\dot{\phi}^2 -2l\dot{\phi}\left( \dot{x}\cos{\phi}+\dot{z}\sin{\phi}\right)
$$

### Energy

<table>
<tr><td></td><th>Kinetic Energy</th><th>Potential Energy</th></tr>
<tr><th>pvtol</th>
<td>$$ T_v = \frac{1}{2} Mv^2 + \frac{1}{2} J \dot{\theta}^2$$</td>
<td>$$ V_v = Mgz$$</td></tr>
<tr><th>pole</th>
<td>$$ T_p = \frac{1}{2} mv_p^2 + \frac{1}{2} j \dot{\phi}^2$$</td>
<td>$$ V_p = mgz_p$$</td></tr>
</table>


### Lagrangian

$$
L = T - V
$$

$$
L = \frac{1}{2}Mv^2 +  \frac{1}{2} J \dot{\theta}^2 +\frac{1}{2}mv_p^2 + \frac{1}{2} j \dot{\phi}^2 -  Mgz - mgz_p
$$

<!--
$$
\begin{multline}
L = \frac{1}{2}M\left( \dot{x}^2 + \dot{z}^2 \right)
    + \frac{1}{2} J \dot{\theta}^2 \\
	+ \frac{1}{2}m\left( \left( \dot{x}^2 + \dot{z}^2 \right) + l^2\dot{\phi}^2 -2l\dot{\phi}\left( \dot{x}\cos{\phi}+\dot{z}\sin{\phi}\right) \right)
	+ \frac{1}{2} j \dot{\phi}^2 \\
	-  Mgz 
	- mg \left(z + l \cos{\phi}\right)
\end{multline}
$$
-->

Expanding with state variables, we get:

$$
\begin{multline}
L = \frac{1}{2} \left( M+m \right)  \left( \dot{x}^2 + \dot{z}^2 \right) + \frac{1}{2} J \dot{\theta}^2 + \frac{1}{2} j \dot{\phi}^2 \\
+ \frac{1}{2}m\left( l^2\dot{\phi}^2 -2l\dot{\phi}\left( \dot{x}\cos{\phi}+\dot{z}\sin{\phi}\right) \right)
- \left( M+m \right) g z - mgl \cos{\phi}
\end{multline}
$$

Partial derivatives of the Lagrangian with respect to the state vector components are:

$$
\begin{cases}
\frac{\partial{L}}{\partial{x}} =  0 \\
\frac{\partial{L}}{\partial{z}} =  - \left( M+m \right) g\\
\frac{\partial{L}}{\partial{\theta}} = 0 \\ 
\frac{\partial{L}}{\partial{\phi}} = ml \left( \dot{\phi} \left( \dot{x}\sin{\phi} - \dot{z}\cos{\phi}\right) + g \sin{\phi}\right)\\
\frac{\partial{L}}{\partial{\dot{x}}} =  \left( M+m \right) \dot{x} - ml\dot{\phi}\cos{\phi}\\
\frac{\partial{L}}{\partial{\dot{z}}} =  \left( M+m \right) \dot{z} - ml\dot{\phi}\sin{\phi}\\
\frac{\partial{L}}{\partial{\dot{\theta}}} = J \dot{\theta}\\ 
\frac{\partial{L}}{\partial{\dot{\phi}}} = j \dot{\phi} + ml \left( l\dot{\phi} - \left( \dot{x}\cos{\phi}+\dot{z}\sin{\phi} \right)\right)\\
\end{cases}
$$
