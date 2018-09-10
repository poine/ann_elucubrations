import sys
from gym.envs.registration import register
################################################################################
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
# ROS version
if sys.version_info[0] < 3:
    register(
        id='pendulum-ros-v0',
        entry_point='gym_foo.envs:PendulumRosEnv',
    )

################################################################################
## Double Pendulum
##
if sys.version_info[0] < 3:
    register(
        id='doublependulum-ros-v0',
        entry_point='gym_foo.envs:DoublePendulumRosEnv',
    )


################################################################################
## CartPole
##
# modification of the original gym code for continuous input
register(
    id='cont-cartpole-v0',
    entry_point='gym_foo.envs:CartPoleEnv',
)
# added a position input driving a reference model
register(
    id='cont-cartpole-with-ref-v0',
    entry_point='gym_foo.envs:CartPoleWithRefEnv',
)
# Random reset for learning swing up
register(
    id='cont-cartpole-v1',
    entry_point='gym_foo.envs:CartPoleUpEnv',
)
# ROS version
if sys.version_info[0] < 3:
    register(
        id='cartpole-ros-v0',
        entry_point='gym_foo.envs:CartPoleRosEnv',
    )
    register(
        id='cartpole-ros-v1',
        entry_point='gym_foo.envs:CartPoleUpRosEnv',
    )


################################################################################
## PVTOL
##
register(
    id='planar_quad-v0',
    entry_point='gym_foo.envs:PlanarQuadEnv',
)


################################################################################
## PVTOL POLE
##
register(
    id='pvtol_pole-v0',
    entry_point='gym_foo.envs:PVTOLPoleEnv',
)


################################################################################
## Bicycle
##
register(
    id='bicycle-v0',
    entry_point='gym_foo.envs:BicycleEnv',
)
register(
    id='bicycle-v1',
    entry_point='gym_foo.envs:BicycleEnv1',
)
register(
    id='bicycle-v2',
    entry_point='gym_foo.envs:BicycleEnv2',
)
# ROS version (Julie Golfcart)
if sys.version_info[0] < 3:
    register(
        id='julie-v0',
        entry_point='gym_foo.envs:JulieEnv',
    )

