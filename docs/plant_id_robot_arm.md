---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Robot Arm Plant Identification


### Full State

$$
\begin{pmatrix} \theta \\ \dot{\theta} \end{pmatrix}_{k+1} = f\left( \theta_k,  \dot{\theta}_k, u_k \right)
$$

[code](https://github.com/poine/ann_elucubrations/blob/master/src/plant_id__robot_arm__fs.py)

<figure>
  <img src="images/plant_id__robot_arm__fs.png" alt="Robot Arm Trajectory">
  <figcaption>Fig1. - Robot Arm trajectory.</figcaption>
</figure>


### Input/Output

$$
\theta_{k+1} = f\left( \theta_k,  \theta_{k-1}, u_k, u_{k-1} \right)
$$

<figure>
  <img src="images/plant_id__robot_arm__io.png" alt="Robot Arm Trajectory">
  <figcaption>Fig2. - Robot Arm trajectory.</figcaption>
</figure>
