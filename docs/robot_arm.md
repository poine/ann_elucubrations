---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Robot Arm

### Description
The single joint `Robot Arm` (or pendulum) is a classic example used in control theory. It consists in a solid rotating around a fixed axis. It is controlled by a torque applied to the rotation axis. It's a mildly non linear second order system.

<figure>
  <img src="images/kuka-robotic-arm.jpg" width="256" alt="Kuka Robot Arm">
  <img src="drawings/robot_arm.png" width="252" alt="Robot Arm Schematics">
  <figcaption>Fig1. - Robot Arm Picture (left) and Schematics (right).</figcaption>
</figure>


### Model

$$
X = \begin{pmatrix} \theta \\ \dot{\theta}\end{pmatrix} \quad U = \begin{pmatrix} \tau\end{pmatrix}
$$

Using Physical parameters, the dynamic can be written as:

$$
 \frac{d}{dt}\begin{pmatrix} \theta \\ \dot{\theta} \end{pmatrix} = 
 \begin{pmatrix}  \dot{\theta} \\ \frac{1}{J} \left( -m.g.l.\sin{\theta} - b.\dot{\theta} + \tau \right) \end{pmatrix}
$$

The dynamic can be abstractly rewritten as 

$$
 \frac{d}{dt}\begin{pmatrix} \theta \\ \dot{\theta} \end{pmatrix} = 
 \begin{pmatrix}  \dot{\theta} \\ -a.\sin{\theta} -b.\dot{\theta} + c.\tau \end{pmatrix}
$$


Figure (2) shows a simulation of the model, obtained with this [code](https://github.com/poine/ann_elucubrations/blob/master/src/robot_arm.py).

<figure>
  <img src="images/robot_arm_free_trajectory.png" alt="Robot Arm Trajectory">
  <figcaption>Fig2. - Robot Arm trajectory.</figcaption>
</figure>
