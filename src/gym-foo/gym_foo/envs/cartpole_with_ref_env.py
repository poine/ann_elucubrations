import gym, gym.utils.seeding, math, numpy as np
#from gym import error, spaces, utils
#from gym.utils import seeding
# adapted from /home/poine/.local/lib/python2.7/site-packages/gym/envs/classic_control/cartpole.py

import misc
import pdb

class CartPoleWithRefEnv(gym.Env):
    metadata = {'render.modes': ['human']}
  
    def __init__(self):
        self.dt, self.max_steps = 0.01, 2000
        
        self.cartpole = misc.CartPole()
        self.ref = misc.SecOrdLinRef(omega=1, xi=0.7)
        
        self.theta_threshold = np.deg2rad(20)
        self.x_threshold = 2.4

        high = np.array([
            self.x_threshold * 2,      # x cart
            np.finfo(np.float32).max,  # xdot cart
            self.theta_threshold * 2,  # theta cart
            np.finfo(np.float32).max,  # thetadot cart
            self.x_threshold,          # x ref
            np.finfo(np.float32).max,  # xdot ref
            np.finfo(np.float32).max,  # xdotdot ref
        ])
        
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32)
        
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    
    def step(self, action):
        # take action
        #print(action)

        self.step_no += 1
        self.cartpole.run(self.dt, action[0])
        self.ref.run(self.dt, self.setpoints[self.step_no])

        #dX = np.array([])
        x, xd, theta, theta_dot = self.cartpole.state
        xr, xrd, xrdd = self.ref.X
        dx, dxd = x-xr, xd-xrd
        
        #cost = -0.5 + 0.1*(dx**2) + 0.02*(dxd**2) + 0.125*(theta**2) + 0.01*(theta_dot**2) + 0.01*(action[0]**2)
        cost = -0.5 + 0.2*(dx**2) + 0.01*(action[0]**2)
        reward = -cost

        over =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold \
                or theta > self.theta_threshold
        info = {}
        state = np.concatenate((self.cartpole.state, self.ref.X))
        return state, reward, over, info
        
    def reset(self):
        self.cartpole.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.ref.X[0] = self.cartpole.state[0]
        self.ref.X[1] = self.cartpole.state[1]
        self.step_no = 0
        _unused, self.setpoints = misc.make_random_pulses(self.dt, self.max_steps, min_nperiod=10, max_nperiod=100, min_intensity=-2, max_intensity=2.)
        return np.concatenate((self.cartpole.state, self.ref.X))
    
  
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

            self.reftrans = rendering.Transform()
            ref = rendering.Line((0,0), (0,carty))
            ref.set_color(.8,.1,.1)
            ref.add_attr(self.reftrans)
            self.viewer.add_geom(ref)
               
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.cartpole.state is None: return None
        x, theta = self.cartpole.state[0], self.cartpole.state[2]
        cartx = x*scale+screen_width/2.0 # MIDDLE OF CART
        refx = self.ref.X[0]*scale+screen_width/2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)
        self.reftrans.set_translation(refx, 0.)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
