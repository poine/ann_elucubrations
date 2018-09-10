import gym, gym.utils.seeding, math, numpy as np


from . import misc
import pdb

class PlanarQuadEnv(gym.Env):
    
    def __init__(self):
        self.dt, self.max_steps = 0.01, 2000
        self.pvtol = misc.PVTOL()

        self.setpoint = np.array([0, 0])
        
        self.x_threshold = 2.4
        self.z_threshold = 1.8
        self.theta_threshold = np.deg2rad(45)
        
        high = np.array([
            self.x_threshold * 2,      # x
            self.z_threshold * 2,      # z
            self.theta_threshold * 2,  # theta
            np.finfo(np.float32).max,  # xdot
            np.finfo(np.float32).max,  # zdot
            np.finfo(np.float32).max,  # theta dot
            self.x_threshold * 2,      # x sp
            self.z_threshold * 2       # z sp
        ])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        f_max = 2*self.pvtol.mass*self.pvtol.gravity
        u_low, u_high = np.array([-f_max, -f_max]), np.array([f_max, f_max])
        self.action_space = gym.spaces.Box(low=u_low, high=u_high, dtype=np.float32)

        self.viewer = None
        self.state = None
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.step_no += 1
        x, z, th, xd, zd, thd = self.pvtol.run(self.dt, action[0], action[1])
        dx, dz = x-self.setpoint[0], z-self.setpoint[1]
        #cost = -0.5 + 0.1*(dx*dx + dz*dz) + 0.05*(xd*xd + zd*zd) + 0.0125*np.linalg.norm(action)
        Q, R = np.diag([0.2, 0.2, 0., 0.05, 0.05, 0.04]), np.diag([0.0125, 0.0125])
        X, Xr = self.pvtol.state, np.array([self.setpoint[0], self.setpoint[1], 0, 0, 0, 0])
        dX = X - Xr
        #cost = np.dot(np.dot(X.T, Q), X)
        #pdb.set_trace()
        cost = -0.5 + np.dot(np.dot(X.T, Q), X) + np.dot(np.dot(action.T, R), action)
        reward = -cost

        over =     x < -self.x_threshold \
               or  x > self.x_threshold  \
               or  z < -self.z_threshold \
               or  z > self.z_threshold  \
               or th < -self.theta_threshold \
               or th >  self.theta_threshold

        info = {}

        return self. _get_state(), reward, over, info

    def reset(self):
        l = 0.9*np.array([self.x_threshold, self.z_threshold, self.theta_threshold, 0.5, 0.5, 0.1])
        self.pvtol.state = self.np_random.uniform(low = -l, high=l)
        #pdb.set_trace()
        #self.pvtol.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.step_no = 0
        return self. _get_state()


    def _get_state(self): return np.concatenate((self.pvtol.state, self.setpoint))


    def render(self, mode='human', close=False):
        screen_width, screen_height = 600, 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width

        quadwidth, quadheight = 50, 5
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -quadwidth/2, quadwidth/2, quadheight/2, -quadheight/2
            quad = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.quadtrans = rendering.Transform()
            quad.add_attr(self.quadtrans)
            self.viewer.add_geom(quad)

            self.reftrans = rendering.Transform()
            self.refcircle = rendering.make_circle(quadheight/2)
            self.refcircle.add_attr(self.reftrans)
            self.refcircle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.refcircle)
            
        if self.pvtol.state is None: return None

        x, z, th, xd, zd, thd = self.pvtol.state
        px = x*scale+screen_width/2.0  # middle of quad
        py = z*scale+screen_height/2.0 # middle of quad
        self.quadtrans.set_translation(px, py)
        self.quadtrans.set_rotation(th)
        rx, rz = self.setpoint
        px = rx*scale+screen_width/2.0  # ref
        py = rz*scale+screen_height/2.0 # ref
        self.reftrans.set_translation(px, py)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
