import time, gym, gym.utils.seeding, math, numpy as np
#from gym import error, spaces, utils
#from gym.utils import seeding
# adapted from /home/poine/.local/lib/python2.7/site-packages/gym/envs/classic_control/cartpole.py

from . import misc
from . import cartpole

class CartPoleEnv(gym.Env):
    metadata = {'render.modes': ['human']}
  
    def __init__(self):

        self.dt = 0.01
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)

        self.theta_threshold = np.deg2rad(12)
        self.x_threshold = 2.4

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold * 2,
            np.finfo(np.float32).max])
        
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32)

        
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _dyn(self, action):
        x, x_dot, theta, theta_dot = self.state
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (action[0] + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        return np.array([x, x_dot, theta, theta_dot])
    
    def step(self, action):
        # take action
        self.state = self._dyn(action)
        x, x_dot, theta, theta_dot = self.state
        cost = -0.5 + 0.1*(x**2) + 0.5*(theta**2) + 0.1*(theta_dot**2) + 0.01*(action[0]**2)
        reward = -cost

        over =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold \
                or theta > self.theta_threshold
        info = {}
        return self.state, reward, over, info
        
    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)
    
  
    def render(self, mode='human', close=False):
        #print self.t, self.x, self.xd
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None
        x, theta = self.state[0], self.state[2]
        cartx = x*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')



class CartPoleUpEnv(gym.Env):

    def __init__(self, max_force=10.):
        self.dt = 0.01
        self.x_threshold = 2.4
        self.max_step = 1000
        obs_high = np.array([
            self.x_threshold * 2,       # x
            np.finfo(np.float32).max,   # xd
            1,                          # cos theta
            1,                          # sin theta
            np.finfo(np.float32).max])  # theta_dot
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-max_force, high=max_force, shape=(1,), dtype=np.float32)
        self.dyn = cartpole.Dynamics()
        self.renderer = None

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.dyn.state
        cth, sth = np.cos(theta), np.sin(theta)
        return np.array([x, x_dot, cth, sth, theta_dot])
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        x_margin = 0.2*self.x_threshold; x0_max = self.x_threshold-x_margin 
        x0 = self.np_random.uniform(low=-x0_max, high=x0_max, size=(1,))[0]
        xd0 = self.np_random.uniform(low=-0.5, high=0.5, size=(1,))[0]
        if 0: # start from any angle - this seems to be bad for learning to swing up
            th0 = self.np_random.uniform(low=-np.pi, high=np.pi, size=(1,))[0]
        else:
            th_min = np.pi/2
            th0 = self.np_random.uniform(low=th_min, high=np.pi, size=(1,))[0]
            sign = 1 if self.np_random.randint(2) else -1
            th0 *= sign
        thd0 = self.np_random.uniform(low=-0.1, high=0.1, size=(1,))[0]
        self.dyn.reset([x0, xd0, th0, thd0])
        #self.last_render_time = time.time()
        self.nb_step = 0
        return self._get_obs()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.dyn.run(self.dt, action)
        self.nb_step += 1
        cost_dist   = 0.1*(x**2)
        cost_angle  = 0.5*(misc.angle_normalize(theta)**2)
        cost_action = 0.01*(action[0]**2)
        reward = 1. - cost_dist - cost_angle - cost_action
        over =  abs(x) > self.x_threshold or self.nb_step > self.max_step
        info = {}
        return self._get_obs(), reward, over, info

    def render(self, mode='human', fps=25.):
        if self.renderer is None:
            self.renderer = cartpole.Rendering(world_width=2.5*self.x_threshold)
        #_now = time.time()
        #_dt = _now - self.last_render_time
        #if _dt >= 1./fps:
        #    self.last_render_time += 1./fps
        return self.renderer.render(self.dyn.state[0], self.dyn.state[2], mode)
        #else:
        #    return None
