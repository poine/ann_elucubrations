'''

   openai gym bicycle environment

'''
import gym, gym.utils.seeding, math, numpy as np

import os

import bicycle_dynamics as bcd, two_d_guidance as tdg
from . import misc
from . import bicycle_utils as bcu
import pdb

class BicycleEnv(gym.Env):

    def __init__(self):
        self.dt, self.max_steps = 0.01, 2000
        self.bicycle = bcd.Plant()
        self.viewer = None
        self.X = np.zeros(bcd.s_size)

    def load_config(self, cfg):
        print('in bicycle env:: load config {}'.format(cfg))
        self.reset_at_random_track_location = cfg['reset_at_random_track_location']
        self.paths = [tdg.path.Path(load=_filename) for _filename in cfg['path_filenames']]

        self.carrot_dists = cfg['carrot_dists']
        self.v_sp = cfg['vel_sp']
      
        self.cfg = cfg
        self.bicycle.P.m  = cfg['car_m']
        self.bicycle.P.J  = cfg['car_j']
        self.bicycle.P.Lf = cfg['car_lf']
        self.bicycle.P.Lr = cfg['car_lr']
        self.bicycle.P.mu = cfg['car_mu']
        self.bicycle.P.compute_aux()

        self.err_track_max = cfg['err_track_max']
        self.err_heading_max = cfg['err_heading_max']

        self.cost_tracking_err =  cfg['cost_tracking_err']
        self.cost_heading_err =  cfg['cost_heading_err']
        self.cost_steering =  cfg['cost_steering']
        self.cost_dsteering =  cfg['cost_dsteering']
        self.cost_vel  = cfg['cost_vel']
        self.cost_dthrottle  = cfg['cost_dthrottle']

        self.reward_dist = cfg['reward_dist']

        self.steering_only = cfg['steering_only']
        
        self.setup_obs_and_act()

    def setup_obs_and_act(self):
        print('in BicycleEnv::setup_obs_and_act')
        # Observation
        obs_hight = np.array([100]*2*(len(self.carrot_dists)+1)) # p0..pn
        self.observation_space = gym.spaces.Box(-obs_hight, obs_hight, dtype=np.float32)

        # Action
        #self.steering_only = True 
        if self.steering_only:
            u_low, u_hight = np.array([-self.bicycle.P.steering_max]), np.array([self.bicycle.P.steering_max])
        else:
            u_low =   np.array([-1., -self.bicycle.P.steering_max])
            u_hight = np.array([ 1.,  self.bicycle.P.steering_max])
        self.action_space = gym.spaces.Box(low=u_low, high=u_hight, dtype=np.float32)

        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    
    def _get_state(self):
        return self.carrots_b
        #return np.append(self.carrots_b.flatten(), [self.X[bcd.s_vy], self.X[bcd.s_psid]])
        #return np.append(self.carrots_b, [self.X[bcd.s_vx], self.X[bcd.s_vy], self.X[bcd.s_psid]])

    def _compute_state(self):
        p0_w, psi = self.X[bcd.s_x:bcd.s_y+1], self.X[bcd.s_psi]
        cy, sy = np.cos(psi), np.sin(psi)
        w2b = np.array([[cy, sy],[-sy, cy]])
        self.carrot_idxs, end_reached = self.path.find_carrots_looped(p0_w, self.carrot_dists)
        self.carrots_w = np.array(self.path.points[self.carrot_idxs])
        self.carrots_b = np.array([np.dot(w2b, p-p0_w) for p in self.carrots_w])
        self.curvatures = np.array(self.path.curvatures[self.carrot_idxs])
        # vel
        self.v = np.linalg.norm(self.X[bcd.s_vx:bcd.s_vy+1])
        # vel angle
        #if abs(self.v) > 0.2:
        self.beta = np.arctan2(self.X[bcd.s_vy], self.X[bcd.s_vx])
        #else:
        #    self.beta = 0
        # carrot angles
        self.gamma = np.arctan2([self.carrots_b[1:, 1]], [self.carrots_b[1:, 0]])
        if end_reached:
            self.nb_laps += 1
            self.saved_dist += self.path.dists[-1]-self.path.dists[self.idx0]
            self.idx0 = 0
        self.total_dist = self.saved_dist+(self.path.dists[self.carrot_idxs[0]]-self.path.dists[self.idx0])
        self.err_tracking = np.linalg.norm(self.carrots_b[0,:])
        self.err_heading = misc.norm_angle(self.X[bcd.s_psi] - self.path.headings[self.carrot_idxs[0]])
        #self.err_heading = misc.norm_angle(self.X[bcd.s_psi] + self.beta - self.path.headings[self.carrot_idxs[0]])
        #pdb.set_trace()

        
    def reset(self, X0=None):
        self.idx_path = self.np_random.randint(len(self.paths))
        print('path {}'.format(self.idx_path))
        self.path = self.paths[self.idx_path]
        if X0 is not None:
            x0, y0, psi0 = X0
            self.X = np.array([x0, y0, psi0, 0, 0, 0])
            self.idx0, _unused = self.path.find_closest([x0, y0], len(self.path.points))
        else:
            self.idx0 = self.np_random.randint(len(self.path.points))
            self.path.last_passed_idx = self.idx0
            x0, y0, psi0 = self.path.points[self.idx0,0], self.path.points[self.idx0,1], self.path.headings[self.idx0]
            l = np.array([0.1, 0.1, np.deg2rad(20)])
            dx0, dy0, dpsi0 = self.np_random.uniform(low = -l, high=l)
            v0 = np.random.uniform(low=0, high=1.2*self.v_sp)
            self.X = np.array([x0+dx0, y0+dy0, psi0+dpsi0, v0, 0, 0])
        #self.path.reset()
        self.saved_dist, self.nb_laps = 0., 0 # dist from previous laps
        self.action = np.array([0,0])
        self._compute_state()
        
        return self._get_state()

    def step(self, action):
        #print(action)
        if len(action) < 2: # with steering only
            vel = np.linalg.norm(self.X[bcd.s_vx:bcd.s_vx+1])
            accel = -2*(vel-self.v_sp)
            action = [accel, action[0]]
        self.X =  self.bicycle.disc_dyn(self.X, action, self.dt)
        self._compute_state()
        _cost_heading  = self.cost_heading_err * np.abs(self.err_heading)
        _cost_tracking = self.cost_tracking_err * self.err_tracking      
        _cost_steering = self.cost_steering * abs(action[bcd.i_df])
        _cost_dsteering = self.cost_dsteering * abs( action[bcd.i_df] - self.action[bcd.i_df])
        _cost_dthrottle = self.cost_dthrottle * abs( action[bcd.i_a] - self.action[bcd.i_a])
        _cost_vel = self.cost_vel * np.abs(self.v_sp - self.v)
        _reward_dur = 1. #  if self.v > 0.1 else 0
        _reward_drift = 0.00#*np.abs(self.X[bcd.s_vy])
        _reward_dist = self.reward_dist * (self.total_dist) # was 0.01
        reward = _reward_dur + _reward_drift + _reward_dur + _reward_dist - _cost_heading - _cost_tracking \
                 - _cost_steering - _cost_dsteering - _cost_dthrottle - _cost_vel 

        _failed_track = self.err_tracking >= self.err_track_max #0.4 # was .25
        _failed_heading =  np.abs(self.beta) > np.pi/2 or np.abs(self.err_heading) > self.err_heading_max
        over =  _failed_track or _failed_heading
        if over:
            print('dist {} / err track {} / err heading {}'.format(self.total_dist, _failed_track, _failed_heading))
        info = {}
        self.action = action # save action for rendering and subsequent step
        return self. _get_state(), reward, over, info


                   
    def render(self, mode='human', close=False, info=None):
        if self.viewer is None:
            #screen_width, screen_height, cockpit_height = 800, 400, 200
            screen_width, screen_height, cockpit_height = 1600, 800, 400
            self.viewer = bcu.BicycleViewer(screen_width, screen_height, cockpit_height, self.paths, len(self.carrot_dists), self.cfg)

        x, y, psi = self.X[bcd.s_x], self.X[bcd.s_y], self.X[bcd.s_psi]
        throttle, alpha = self.action
 
        res = self.viewer.render(x, y, psi, self.v, alpha, throttle, self.carrots_b, mode)
        return res

class BicycleEnv1(BicycleEnv):
    
    ''' This version is for comparing different observation configurations '''

    def load_config(self, cfg):
        self.obs_cfg_ixd = cfg['obs_cfg']
        BicycleEnv.load_config(self, cfg)
    
    def setup_obs_and_act(self):
        print('in BicycleEnv1::setup_obs_and_act')
        self.obs_cfg = [
            ('0: p1_y',
             [10],     lambda:  [self.carrots_b[1, 1]]),
            ('1: p0_y, p1_y',
             [10, 10], lambda:  [self.carrots_b[0, 1], self.carrots_b[1, 1]]),
            ('2: p0_y, p1_angle',
             [10, 10], lambda:  [self.carrots_b[0, 1], self.gamma]),
            ('3: p0_y, p1_y, beta',
             [10, 10, 2], lambda: [self.carrots_b[0, 1], self.carrots_b[1, 1], self.beta]),
            ('4: p0_y, p1_angle, beta',
             [10, 10, 2], lambda: [self.carrots_b[0, 1], self.gamma, self.beta]),
            ('5: p0_y, p1_y, c0',
             [10, 10, 2], lambda: [self.carrots_b[0, 1], self.carrots_b[1, 1], self.curvatures[0]]),
            ('6: p0_x, p0_y, p1_x, p1_y',
             [10, 10, 10, 10],
             lambda: [self.carrots_b[0, 0], self.carrots_b[0, 1], self.carrots_b[1, 0], self.carrots_b[1, 1]]),
            ('7: p0_y, p1_x, p1_y',
             [10, 10, 10],
             lambda: [self.carrots_b[0, 1], self.carrots_b[1, 0], self.carrots_b[1, 1]]),
            ('8: p0_y, p1_x, p1_y, c0',
             [10, 10, 10, 2],
             lambda: [self.carrots_b[0, 1], self.carrots_b[1, 0], self.carrots_b[1, 1], self.curvatures[0]]),
            ('9: p0_y, p1_x, p1_y, c0, beta, psid',
             [10, 10, 10, 2, 1, 5],
             lambda: [self.carrots_b[0, 1], self.carrots_b[1, 0], self.carrots_b[1, 1], self.curvatures[0], self.beta, self.X[bcd.s_psid]]),
            ('10: p0_y, p1_x, p1_y, c0, c1, beta, psid',
             [10, 10, 10, 2, 2, 1, 5],
             lambda: [self.carrots_b[0, 1], self.carrots_b[1, 0], self.carrots_b[1, 1], self.curvatures[0], self.curvatures[1], self.beta, self.X[bcd.s_psid]]),
            ('11: p0_y, p1_x, p1_y, c0, c1, c2, beta, psid',
             [10, 10, 10, 2, 2, 2, 1, 5],
             lambda: [self.carrots_b[0, 1], self.carrots_b[1, 0], self.carrots_b[1, 1], self.curvatures[0], self.curvatures[1], self.curvatures[2], self.beta, self.X[bcd.s_psid]]),
            ('12: p0_y, p1_x, p1_y, c0, c1, c2, beta, psid, v',
             [10, 10, 10, 2, 2, 2, 1, 5, 5],
             lambda: [self.carrots_b[0, 1], self.carrots_b[1, 0], self.carrots_b[1, 1], self.curvatures[0], self.curvatures[1], self.curvatures[2], self.beta, self.X[bcd.s_psid], self.v]),
            ('13: p0_y, p1_x, p1_y, c0, c1, c2, c3, beta, psid, v',
             [10, 10, 10, 2, 2, 2, 2, 1, 5, 5],
             lambda: [self.carrots_b[0, 1], self.carrots_b[1, 0], self.carrots_b[1, 1], self.curvatures[0], self.curvatures[1], self.curvatures[2], self.curvatures[3], self.beta, self.X[bcd.s_psid], self.v]),
        ]
        print('bicycle1: obs configuration: {}'.format(self.obs_cfg[self.obs_cfg_ixd][0]))
        
        obs_hight = np.array(self.obs_cfg[self.obs_cfg_ixd][1])
        self.observation_space = gym.spaces.Box(-obs_hight, obs_hight, dtype=np.float32)
        # Action
        #self.steering_only = True 
        if self.steering_only:
            u_low, u_hight = np.array([-np.deg2rad(30)]), np.array([np.deg2rad(30)])
        else:
            u_low, u_hight = np.array([-1, -np.deg2rad(30)]), np.array([1, np.deg2rad(30)])
        self.action_space = gym.spaces.Box(low=u_low, high=u_hight, dtype=np.float32)

    def _get_state(self):
        _s = np.array(self.obs_cfg[self.obs_cfg_ixd][2]())
        #print(_s)
        return _s


class BicycleEnv2(BicycleEnv1):
    '''This version is for controling throttle '''

    # def load_config(self, cfg):
    #     print('#### {}'.format(cfg['path_filenames']))
    #     self.paths = [tdg.path.Path(load=_filename) for _filename in cfg['path_filenames']]
    #     cfg['path_filename'] = cfg['path_filenames'][0]
    #     BicycleEnv1.load_config(self, cfg)
    #     self.paths = [tdg.path.Path(load=_filename) for _filename in cfg['path_filenames']]

    # def reset(self, X0=None):
    #     self.idx_path = self.np_random.randint(len(self.paths))
    #     print('path {}'.format(self.idx_path))
    #     self.path = self.paths[self.idx_path]
    #     return BicycleEnv1.reset(self, X0)
    pass
