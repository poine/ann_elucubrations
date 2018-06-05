from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_foo.envs:FooEnv',
)

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
