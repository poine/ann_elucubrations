'''

   openai gym gazebo/ros car

'''
import gym, gym.utils.seeding, math, numpy as np
import os, threading, time
import rospy, std_msgs.msg, nav_msgs.msg, ackermann_msgs.msg, gazebo_msgs.msg, std_srvs.srv, gazebo_msgs.srv
import tf.transformations, pdb

import two_d_guidance as tdg
from . import misc
from . import bicycle_utils as bcu

class JulieEnv(gym.Env):

    def __init__(self, dt=0.02):
        self.dt = dt
        steering_only = True 
        if steering_only:
            u_low, u_hight = np.array([-np.deg2rad(30)]), np.array([np.deg2rad(30)])
        else:
            u_low, u_hight = np.array([-1, -np.deg2rad(30)]), np.array([1, np.deg2rad(30)])
        self.action_space = gym.spaces.Box(low=u_low, high=u_hight, dtype=np.float32)
        self.viewer = None
        self.ros_proxy = RosProxy()

    def load_config(self, cfg):
        self.cfg = cfg
        self.path_filename = cfg['path_filename']
        self.path = tdg.path.Path(load=self.path_filename)
        self.carrot_dists = cfg['carrot_dists']
        self.v_sp = cfg['vel_sp']

        obs_hight = np.array([100]*2*(len(self.carrot_dists)+1))# p0..pn
        self.observation_space = gym.spaces.Box(-obs_hight, obs_hight, dtype=np.float32)

        self.err_track_max = cfg['err_track_max']
        self.err_heading_max = cfg['err_heading_max']

        self.cost_tracking_err =  cfg['cost_tracking_err']
        self.cost_heading_err =  cfg['cost_heading_err']
        self.cost_steering =  cfg['cost_steering']

        self.reward_vel  = cfg['reward_vel']
        self.reward_dist = cfg['reward_dist']
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _compute_state(self):
        p0_w, psi = self.X[0:2], self.X[2]
        cy, sy = np.cos(psi), np.sin(psi)
        w2b = np.array([[cy, sy],[-sy, cy]])
        self.carrot_idxs, end_reached = self.path.find_carrots_looped(p0_w, self.carrot_dists)
        self.carrots_w = np.array(self.path.points[self.carrot_idxs])
        self.carrots_b = np.array([np.dot(w2b, p-p0_w) for p in self.carrots_w])
        if end_reached:
            self.nb_laps += 1
            self.saved_dist += self.path.dists[-1]-self.path.dists[self.idx0]
            self.idx0 = 0
        self.total_dist = self.saved_dist+(self.path.dists[self.carrot_idxs[0]]-self.path.dists[self.idx0])
        self.err_tracking = np.linalg.norm(self.carrots_b[0,:])
        self.err_heading = misc.norm_angle(psi - self.path.headings[self.carrot_idxs[0]])
    
    def _get_obs(self):
        return self.carrots_b
    
    def reset(self, X0=None):
        if X0 is None:
            self.idx0 = self.np_random.randint(len(self.path.points))
            self.path.last_passed_idx = self.idx0
            x0, y0, psi0 = self.path.points[self.idx0,0], self.path.points[self.idx0,1], self.path.headings[self.idx0]
            l = np.array([1., 1., np.deg2rad(20)])
            dx0, dy0, dpsi0 = self.np_random.uniform(low = -l, high=l)
            x0 += dx0; y0 += dy0; psi0 = misc.norm_angle(psi0+dpsi0)
        else:
            x0, y0, psi0 = X0
            self.idx0, _unused = self.path.find_closest([x0, y0], len(self.path.points))
            
        self.X = self.ros_proxy.reset(x0, y0, psi0)
        self._step, self._last_step_time = 0, time.time()
        self.saved_dist, self.nb_laps = 0., 0 # dist from previous laps
        self.sum_sleeps = 0.
        self.action = [0, 0]
        self._compute_state()
        return self._get_obs()
        
    def step(self, action):
        _now = time.time(); _elapsed = _now - self._last_step_time
        sleep_time = self.dt - _elapsed
        if sleep_time > 0: time.sleep(sleep_time)
        self.sum_sleeps += sleep_time
        self._last_step_time += self.dt
        self._step += 1
        if len(action) < 2: # with steering only
            accel = -2*(self.X[3]-self.v_sp)
            self.action = [accel, action[0]]
        self.X =  self.ros_proxy.write_cmd_read_state(alpha=self.action[1], vel=self.action[0])
        self._compute_state()
        #print('err_h:{:.1f} err_t:{:.1f} accel:{:.1f} steering:{:.1f}'.format(self.err_heading, self.err_tracking, self.action[0], self.action[1]))
        
        _cost_heading  = self.cost_heading_err * np.abs(self.err_heading)
        _cost_tracking = self.cost_tracking_err * self.err_tracking
        _cost_steering = self.cost_steering * self.action[1]
        _reward_dur = 1.
        _reward_dist = self.reward_dist * (self.total_dist)
        _reward_vel =  self.reward_vel * (self.X[3])
        reward = _reward_dur + _reward_dist + _reward_vel - _cost_heading - _cost_tracking - _cost_steering
        
        _failed_track = self.err_tracking >= self.err_track_max
        _failed_heading = np.abs(self.err_heading) > self.err_heading_max
        over =  _failed_track or _failed_heading
        if over:
            print('sleep avg {}'.format(self.sum_sleeps/self._step))
        info = {}
        return self._get_obs(), reward, over, info
    
    def render(self, mode='human', close=False):
        screen_width, screen_height, cockpit_height = 800, 400, 200
        w_center, w_extends = self.path.get_extends()
        if self.viewer is None:
            self.viewer = bcu.BicycleViewer(screen_width, screen_height, cockpit_height,
                                            w_center, w_extends, self.path, len(self.carrot_dists), self.cfg)
        x, y, psi, v = self.X[0], self.X[1], self.X[2], self.X[3]
        throttle, steering = self.action
        return self.viewer.render(x, y, psi, v, steering, throttle, self.carrots_b, mode)


    
def list_of_xyz(p): return [p.x, p.y, p.z]
def array_of_xyz(p): return np.array(list_of_xyz(p))
def list_of_xyzw(q): return [q.x, q.y, q.z, q.w] 
def xyzw_of_list(q, l): q.x, q.y, q.z, q.w = l 

class RosProxy:
    def __init__(self, dt=0.01):
        self.dt = dt
        self._alpha, self._vel = 0., 0. 

        self.thread = threading.Thread(target=self._run)
        self.stop = False
        self.lock = threading.Lock()
        rospy.init_node('julie_gym_env', disable_signals=True)
        self.thread.start()

    def reset(self, x0, y0, psi0):
        self.g_pause()
        self.g_reset()
        _msg = gazebo_msgs.msg.ModelState()
        _msg.model_name='julie'
        _msg.pose.position.x, _msg.pose.position.y, _msg.pose.position.z = x0, y0, 0.3
        q = tf.transformations.quaternion_from_euler(0, 0, psi0)
        xyzw_of_list(_msg.pose.orientation, q)
        self.g_set_model_state(_msg)
        self.g_unpause()
        time.sleep(1)
        return np.array([x0, y0, psi0, 0])

    def write_cmd_read_state(self, alpha, vel):
        self.lock.acquire()
        self._alpha, self._vel = alpha, vel
        _state = (self.xy[0], self.xy[1], self.psi, self.v)
        self.lock.release()
        return np.array(_state)
        
    def _run(self):
        # Command
        cmd_topic = '/julie/julie_ackermann_controller/cmd_ack'
        self.pub_cmd = rospy.Publisher(cmd_topic, ackermann_msgs.msg.AckermannDriveStamped, queue_size=1)
        
        # Gazebo control
        self.g_pause = rospy.ServiceProxy("/gazebo/pause_physics", std_srvs.srv.Empty)
        self.g_unpause = rospy.ServiceProxy("/gazebo/unpause_physics", std_srvs.srv.Empty)
        self.g_set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", gazebo_msgs.srv.SetModelState)
        #self.g_set_link_state = rospy.ServiceProxy("/gazebo/set_link_state", gazebo_msgs.srv.SetLinkState)
        #self.g_get_link_state = rospy.ServiceProxy("/gazebo/get_link_state", gazebo_msgs.srv.GetLinkState)
        self.g_reset = rospy.ServiceProxy('/gazebo/reset_simulation', std_srvs.srv.Empty)
        
        # State
        rospy.Subscriber('/julie_gazebo/base_link_truth', nav_msgs.msg.Odometry, self._odom_cbk)
         
        rate = rospy.Rate(1./self.dt)
        while not self.stop:
            try:
                self._publish_action()
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                pass # time goes backward when we reset gazebo


    def _publish_action(self):
        _msg = ackermann_msgs.msg.AckermannDriveStamped()
        _msg.header.stamp = rospy.Time.now()
        _msg.header.frame_id = 'odom'
        _msg.drive.steering_angle = self._alpha
        _msg.drive.speed = self._vel
        self.pub_cmd.publish(_msg)

    def _odom_cbk(self, msg):
        self.lock.acquire()
        self.pose  = msg.pose.pose
        self.ori   = msg.pose.pose.orientation
        self.twist = msg.twist.twist

        self.xy = array_of_xyz(self.pose.position)[:2]
        self.R = tf.transformations.quaternion_matrix(list_of_xyzw(self.pose.orientation))
        self.psi = tf.transformations.euler_from_matrix(self.R)[2]
        self.v = np.linalg.norm(array_of_xyz(self.twist.linear))
        self.psid = self.twist.angular.z
        self.lock.release()
