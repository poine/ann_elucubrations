import sys
# Pendulums
from gym_foo.envs.pendulum_legacy_env import PendulumLegacyEnv
from gym_foo.envs.pendulum_legacy_env import PendulumAngleObsEnv
if sys.version_info[0] < 3:
    from gym_foo.envs.pendulum_ros_env import PendulumRosEnv
    from gym_foo.envs.pendulum_ros_env import DoublePendulumRosEnv

# CartPole
from gym_foo.envs.cartpole_env import CartPoleEnv
from gym_foo.envs.cartpole_env import CartPoleUpEnv
from gym_foo.envs.cartpole_with_ref_env import CartPoleWithRefEnv
if sys.version_info[0] < 3:
    from gym_foo.envs.cartpole_ros_env import CartPoleRosEnv
    from gym_foo.envs.cartpole_ros_env import CartPoleUpRosEnv

# Planar Quad
from gym_foo.envs.planar_quad_env import PlanarQuadEnv
from gym_foo.envs.pvtol_pole_env import PVTOLPoleEnv

# Car
from gym_foo.envs.bicycle_env import BicycleEnv
from gym_foo.envs.bicycle_env import BicycleEnv1
from gym_foo.envs.bicycle_env import BicycleEnv2
if sys.version_info[0] < 3:
    from gym_foo.envs.julie_env import JulieEnv

