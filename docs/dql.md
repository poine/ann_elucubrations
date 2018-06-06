---
title: ANN Elucubrations
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Qlearning is a reinforcement learning method suitable for problems with discrete state space and input (action) space problems. It has been extended to problemswith continuous state space () and then to problems with continuous input state ().

### Q learning:

`Q-Learning attempts to learn the value of being in a given state, and taking a specific action there.`

time difference learning - Bellman Equation

policy gradient


### Simulations

I started from an [implementation](https://github.com/pemami4911/deep-rl/tree/master/ddpg) of  *Deep Deterministic Policy Gradient* (DDPG) by Patrick Emami which had been tested on [OpenAi](https://openai.com/) [Gym](https://gym.openai.com/) [Pendulum](https://gym.openai.com/envs/Pendulum-v0/)

#### Pendulum
the code for the pendulum is [here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py)

The dymaics part reads:
```python
u = np.clip(u, -self.max_torque, self.max_torque)[0]
newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
newth = th + newthdot*dt
newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
```
It looks like

$$
\ddot{\theta} = -\frac{3g}{2l} \sin{(\theta+\pi)} + \frac{3}{ml^2} u
$$

mixed with a first order integration

The cost part reads
```python
costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
```


and the observation reads:
```python
def _get_obs(self):
	theta, thetadot = self.state
	return np.array([np.cos(theta), np.sin(theta), thetadot])
```


`/dql__gym_pendulum.py --env Pendulum-v0 --render-env --train --summary-dir results/run_nonorm`


## References

 * Continuous Control with Deep Reinforcement Learning, Timothy P. Lillicrap et all
 * Deterministic Policy Gradient Algorithms, David Silver et all

 * [Reinforcement Learning: An Introduction, Rich Sutton, MIT Press.](https://mitpress.mit.edu/books/reinforcement-learning)   [online version](http://incompleteideas.net/book/the-book-2nd.html)
  

 * [Reinforcement Learning w/ Keras + OpenAI: Actor-Critic Models](https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69)

 * An in depth lecture [course](http://rll.berkeley.edu/deeprlcourse/)   [Actor-Critic Algorithms](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_5_actor_critic_pdf.pdf)


 * A quick tutorial on priciples [Introduction to Q-Learning](https://towardsdatascience.com/introduction-to-q-learning-88d1c4f2b49c)

 *  Patrick Emami blog post: [Deep Deterministic Policy Gradients in TensorFlow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
