from gym.envs.registration import register

## Pendulum
##
# copy of original gym code
register(
    id='pendulum-legacy-v0',
    entry_point='gym_foo.envs:PendulumLegacyEnv',
)
# modified to use angle as observation
register(
    id='pendulum-ang-obs-v0',
    entry_point='gym_foo.envs:PendulumAngleObsEnv',
)


## CartPole
##
# modification of the original gym code for continuous input
register(
    id='cont-cartpole-v0',
    entry_point='gym_foo.envs:FooEnv',
)
# added a position input driving a reference model
register(
    id='cont-cartpole-with-ref-v0',
    entry_point='gym_foo.envs:CartPoleWithRefEnv',
)

register(
    id='planar_quad-v0',
    entry_point='gym_foo.envs:PlanarQuadEnv',
)

register(
    id='pvtol_pole-v0',
    entry_point='gym_foo.envs:PVTOLPoleEnv',
)
