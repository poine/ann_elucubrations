---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


# Planar VTOL


The following figure describes the variables and parameters of the model:
<figure>
	<img src="drawings/pvtol.svg.png" alt="PVTOL schematics" width="262">
	<figcaption>Fig1. - PVTOL schematics.</figcaption>
</figure>


### Generalized coordinates
 
$$q=\begin{pmatrix}x&z&\theta\end{pmatrix}^T $$

### Kinetic energy

$$ T = \frac{1}{2} m(\dot{x}^2+\dot{z}^2) + \frac{1}{2} J \dot{\theta}^2$$

### Potential energy

$$ V = mgz$$

### Lagrangian

$$\begin{align} 
L &= T-V \\
L &= \frac{1}{2} m(\dot{x}^2+\dot{z}^2) + \frac{1}{2} J \dot{\theta}^2 -  mgz
\end{align}$$


<table>
<tr><td>
$$
\begin{cases}
\frac{\partial{L}}{\partial{x}} =  0 \\
\frac{\partial{L}}{\partial{z}} =  - m g\\
\frac{\partial{L}}{\partial{\theta}} = 0 \\ 
\end{cases}
$$
</td><td>
$$
\begin{cases}
\frac{\partial{L}}{\partial{\dot{x}}} = m\dot{x} \\
\frac{\partial{L}}{\partial{\dot{z}}} = m\dot{z} \\
\frac{\partial{L}}{\partial{\dot{\theta}}} = J \dot{\theta}
\end{cases}
$$
</td></tr>
</table>

### Lagrange Equations

 *
 
$$
\frac{d}{dt}\left( \frac{\partial{L}}{\partial{\dot{x}}} \right) - \frac{\partial{L}}{\partial{x}} = F_x
$$

$$
m\ddot{x} = -(f_1+f_2) \sin{\theta}
$$

 *

$$
\frac{d}{dt}\left( \frac{\partial{L}}{\partial{\dot{z}}} \right) - \frac{\partial{L}}{\partial{z}} = F_z
$$

$$
m\ddot{z} + mg = (f_1+f_2) \cos{\theta}
$$

 *

$$
\frac{d}{dt}\left( \frac{\partial{L}}{\partial{\dot{\theta}}} \right) - \frac{\partial{L}}{\partial{\theta}} = M_{\theta}
$$

$$
J\ddot{\theta} = l \left( -f_1+f_2 \right)
$$

### State Space Representation

$$
X= \begin{pmatrix}x&z&\theta&\dot{x}&\dot{z}&\dot{\theta}\end{pmatrix}^T \qquad
U = \begin{pmatrix}f_1 & f_2 \end{pmatrix}^T
$$

$$
\dot{X} = \begin{pmatrix}
  \dot{x} \\
  \dot{z} \\
  \dot{\theta} \\
  -\frac{1}{m}  \sin{\theta} \left( f_1+f_2 \right) \\
  -g + \frac{1}{m}  \cos{\theta} \left( f_1+f_2 \right)\\
  \frac{l}{J} \left( -f_1+f_2 \right)
\end{pmatrix}
$$

