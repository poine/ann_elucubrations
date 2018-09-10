import gym, gym.utils.seeding, math, numpy as np

from . import pvtol_pole as pvtp

'''

   openai gym pvtol pole environment

'''
import pyglet

class PVTOLPoleEnv(gym.Env):

    def __init__(self):
        self.dt, self.max_steps = 0.01, 2000
        self.pvtp = pvtp.PVTP()

        self.setpoint = np.array([0, 0])

        self.x_threshold, self.z_threshold = 2.4, 1.8
        self.th_threshold, self.ph_threshold = np.deg2rad(45), np.deg2rad(45)
        
        high = np.array([
            self.x_threshold * 2,      # x
            self.z_threshold * 2,      # z
            self.th_threshold * 2,     # theta
            self.ph_threshold * 2,     # phi
            np.finfo(np.float32).max,  # xdot
            np.finfo(np.float32).max,  # zdot
            np.finfo(np.float32).max,  # theta dot
            np.finfo(np.float32).max,  # phi dot
            self.x_threshold * 2,      # x sp
            self.z_threshold * 2       # z sp
        ])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        u_low, u_high = -2*self.pvtp.Ue, 2*self.pvtp.Ue, 
        self.action_space = gym.spaces.Box(low=u_low, high=u_high, dtype=np.float32)

        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def _get_state(self): return np.concatenate((self.pvtp.state, self.setpoint))

    def reset(self, X0=None):
        if X0 is None:
            l = 0.9*np.array([self.x_threshold, self.z_threshold, self.th_threshold, self.ph_threshold, 0.5, 0.5, 0.1, 0.1])
            self.pvtp.state = self.np_random.uniform(low = -l, high=l)
        else:
            self.pvtp.state = X0
        self.step_no = 0
        return self. _get_state()
    
    def step(self, action):
        self.step_no += 1
        X = self.pvtp.disc_dyn(self.pvtp.state, action, self.dt)
        self.pvtp.state = X
        
        Q, R = np.diag([0.2, 0.2, 0., 0., 0.05, 0.05, 0.02, 0.04]), np.diag([0.0125, 0.0125])
        X, Xr = self.pvtp.state, np.array([self.setpoint[0], self.setpoint[1], 0, 0, 0, 0, 0, 0])
        dX = X - Xr
        reward = 0.5 - np.dot(np.dot(X.T, Q), X) - np.dot(np.dot(action.T, R), action)

        over =     X[pvtp.PVTP.s_x]  < -self.x_threshold  \
                or X[pvtp.PVTP.s_x]  >  self.x_threshold  \
                or X[pvtp.PVTP.s_z]  < -self.z_threshold  \
                or X[pvtp.PVTP.s_z]  >  self.z_threshold  \
                or X[pvtp.PVTP.s_th] < -self.th_threshold \
                or X[pvtp.PVTP.s_th] >  self.th_threshold \
                or X[pvtp.PVTP.s_ph] < -self.ph_threshold \
                or X[pvtp.PVTP.s_ph] >  self.ph_threshold

        info = {}

        return self. _get_state(), reward, over, info



    def _create_viewer(self, screen_width, screen_height, quadwidth, quadheight, polewidth, poleheight):
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(screen_width, screen_height)

        l,r,t,b = -quadwidth/2, quadwidth/2, quadheight/2, -quadheight/2
        quad = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        self.quadtrans = rendering.Transform()
        quad.add_attr(self.quadtrans)
        self.viewer.add_geom(quad)

        axleoffset = quadheight/4.0
        l,r,t,b = -polewidth/2,polewidth/2,poleheight-polewidth/2,-polewidth/2
        pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        pole.set_color(.8,.6,.4)
        self.poletrans = rendering.Transform(translation=(0, axleoffset))
        pole.add_attr(self.poletrans)
        pole.add_attr(self.quadtrans)
        self.viewer.add_geom(pole) 
    
    def render(self, mode='human', close=False, info=None):
        screen_width, screen_height = 600, 400
        world_width = self.x_threshold*2
        scale = screen_width/world_width

        if self.viewer is None:
            self._create_viewer(screen_width, screen_height, quadwidth=50, quadheight=5, polewidth=4, poleheight=75)

        x, z, th, ph, xd, zd, thd, phd = self.pvtp.state
        px = x*scale+screen_width/2.0  # middle of quad
        py = z*scale+screen_height/2.0 # middle of quad
        self.quadtrans.set_translation(px, py)
        self.quadtrans.set_rotation(th)
        self.poletrans.set_rotation(ph-th)

        
        res = self.viewer.render(return_rgb_array = mode=='rgb_array')
        if info is not None:
            label = pyglet.text.Label(info,
                                      font_name='Times New Roman',
                                      font_size=36,
                                      x=screen_width/2, y=screen_height/4,
                                      anchor_x='center', anchor_y='center', color=(0,0,0,255))
            label.draw()
        return res
