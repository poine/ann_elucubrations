import gym, gym.utils.seeding, math, numpy as np
#from gym import error, spaces, utils
#from gym.utils import seeding
# adapted from /home/poine/.local/lib/python2.7/site-packages/gym/envs/classic_control/cartpole.py

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

    
    def step(self, action):
        # take action
        #print(action)
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (action[0] + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])

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
