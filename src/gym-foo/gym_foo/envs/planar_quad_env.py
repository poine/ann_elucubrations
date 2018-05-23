import gym, gym.utils.seeding, math, numpy as np


import misc
import pdb

class PlanarQuadEnv(gym.Env):
    
    def __init__(self):
        self.dt, self.max_steps = 0.01, 2000
        self.pvtol = misc.PVTOL()
 
        self.x_threshold = 2.4
        self.z_threshold = 2.4
        self.theta_threshold = np.deg2rad(45)
        
        high = np.array([
            self.x_threshold * 2,      # x
            self.z_threshold * 2,      # z
            self.theta_threshold * 2,  # theta
            np.finfo(np.float32).max,  # xdot
            np.finfo(np.float32).max,  # zdot
            np.finfo(np.float32).max   # theta dot
        ])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        u_low, u_high = np.array([0, 0]), np.array([3,3])
        self.action_space = gym.spaces.Box(low=u_low, high=u_high, dtype=np.float32)

        self.viewer = None
        self.state = None
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.step_no += 1
        x, z, th, xd, zd, thd = self.pvtol.run(self.dt, action[0], action[1])
        
        cost = -0.5 + x + z
        reward = -cost

        over =     x < -self.x_threshold \
               or  x > self.x_threshold  \
               or  z < -self.z_threshold \
               or  z > self.z_threshold  \
               or th < -self.theta_threshold \
               or th >  self.theta_threshold

        info = {}

        state = np.array(self.pvtol.state)
        
        return state, reward, over, info

    def reset(self):
        self.pvtol.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.step_no = 0
        return self.pvtol.state



    def render(self, mode='human', close=False):
        screen_width, screen_height = 600, 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width

        quadwidth, quadheight = 50, 10
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -quadwidth/2, quadwidth/2, quadheight/2, -quadheight/2
            quad = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.quadtrans = rendering.Transform()
            quad.add_attr(self.quadtrans)
            self.viewer.add_geom(quad)
        if self.pvtol.state is None: return None

        x, z, th, xd, zd, thd = self.pvtol.state
        px = x*scale+screen_width/2.0  # middle of quad
        py = z*scale+screen_height/2.0 # middle of quad
        self.quadtrans.set_translation(px, py)
        self.quadtrans.set_rotation(-th)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
