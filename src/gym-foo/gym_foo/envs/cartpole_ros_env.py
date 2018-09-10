'''

   openai gym gazebo/ros cartpole

'''
import gym, gym.utils.seeding, math, numpy as np
import os, threading, time
import rospy, std_msgs.msg, sensor_msgs.msg, gazebo_msgs.msg, std_srvs.srv, gazebo_msgs.srv
import tf.transformations, pdb

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class CartPoleRosEnv(gym.Env):

    def __init__(self):
        self.dt = 0.025
        self.max_force = 10.
        self.action_space = gym.spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)

        self.theta_threshold, self.x_threshold = np.deg2rad(45.), 1.3
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold * 2,
            np.finfo(np.float32).max])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        cmd_topic = '/cartpole/beam_to_cart_effort_controller/command'
        self.ros_proxy = RosProxy('cartpole', cmd_topic, self.dt)
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_observation(self, _p, _v):
        return np.concatenate((_p, _v))
        #pdb.set_trace()
    
    def reset(self):
        x0  = self.np_random.uniform(low=-1., high=1., size=(1,))[0]
        xd0 = self.np_random.uniform(low=-0.5, high=0.5, size=(1,))[0]
        th0 = self.np_random.uniform(low=-np.deg2rad(5.), high=np.deg2rad(5.), size=(1,))[0]
        _p, _v = self.ros_proxy.reset(x0, xd0, th0)
        self._last_step_time = time.time()
        self._step = 0
        #time.sleep(1.)
        #_p, _v = self.ros_proxy.write_cmd_read_state(0)
        return self._get_observation(_p, _v)
        
    def step(self, action):
        _now = time.time()
        _dt = _now - self._last_step_time
        if _dt < self.dt: time.sleep(self.dt-_dt)
        self._last_step_time = _now
        self._step+=1
        _p, _v = self.ros_proxy.write_cmd_read_state(action)

        _cost_pos   = 0.1*(_p[0]**2)
        _cost_angle = 0.5*(_p[1]**2)
        _cost_vel   = 0.02*(_v[1]**2)
        _cost_rvel  = 0.1*(_v[1]**2)
        _cost_act = 0.01*(action[0]**2)
        reward = 1. - _cost_pos - _cost_angle - _cost_act

        #print(_p)
        over =  _p[0] < -self.x_threshold or \
                _p[0] >  self.x_threshold or \
                _p[1] < -self.theta_threshold or \
                _p[1] >  self.theta_threshold

        info = {}

        return self._get_observation(_p, _v), reward, over, info
        
        
    def render(self, mode='human', close=False):
        pass





class CartPoleUpRosEnv(CartPoleRosEnv):
    
    def __init__(self):
        CartPoleRosEnv.__init__(self)
        high = np.array([
            self.x_threshold * 2, 1, 1,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.max_force = 50.
        self.action_space = gym.spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)

    def _get_observation(self, _p, _v):
        sth, cth = np.sin(_p[1]), np.cos(_p[1])
        return np.concatenate(([_p[0], sth, cth], _v))
        
    def reset(self):
        x0  = self.np_random.uniform(low=-.5, high=0.5, size=(1,))[0]
        xd0 = self.np_random.uniform(low=-0.5, high=0.5, size=(1,))[0]
        if 0: # start from any angle - this seems to be bad for learning to swing up
            th0 = self.np_random.uniform(low=-np.pi, high=np.pi, size=(1,))[0]
        else:
            th_min = np.pi/2
            th0 = self.np_random.uniform(low=th_min, high=np.pi, size=(1,))[0]
            sign = 1 if self.np_random.randint(2) else -1
            th0 *= sign
        _p, _v = self.ros_proxy.reset(x0, xd0, th0)
        self._last_step_time = time.time()
        self._step = 0
        return self._get_observation(_p, _v)
        
    def step(self, action):
        _now = time.time()
        _elapsed = _now - self._last_step_time
        if _elapsed < self.dt: time.sleep(self.dt-_elapsed)
        self._last_step_time += self.dt
        self._step += 1
        _p, _v = self.ros_proxy.write_cmd_read_state(action)

        _cost_pos   = 0.1*(_p[0]**2)
        _cost_angle = 0.5*(angle_normalize(_p[1])**2)
        _cost_vel   = 0.02*(_v[0]**2)
        _cost_rvel  = 0.1*(_v[1]**2)
        _cost_act = 0.01*(action[0]**2)
        reward = 1. - _cost_pos - _cost_angle - _cost_vel - _cost_rvel - _cost_act

        over = abs(_p[0]) > self.x_threshold

        info = {}

        return self._get_observation(_p, _v), reward, over, info







    
class RosProxy:
    def __init__(self, name, cmd_topics, dt):
        self._u = 0

        self.name, self.cmd_topics, self.dt = name, cmd_topics, dt
        self.ros_thread = threading.Thread(target=self._run)
        self.ros_thread_stop = False
        self.lock = threading.Lock()
        rospy.init_node('{}_ros_ctl'.format(name), disable_signals=True)
        self.ros_thread.start()

    def reset(self, x0, xd0, th0):
        self.g_pause()
        self.g_reset()
        if 1:
            _msg = gazebo_msgs.msg.LinkState()
            _msg.link_name = 'cart'
            _msg.reference_frame = 'beam'
            _msg.pose.position.x, _msg.pose.position.y, _msg.pose.position.z = x0, 0., 0.125 
            #_msg.pose.orientation.y,_msg.pose.orientation.w = q[1], q[3]
            _msg.pose.orientation.x, _msg.pose.orientation.y, _msg.pose.orientation.z, _msg.pose.orientation.w = 0, 0, 0, 1
            #_msg.twist.angular.x = _msg.twist.angular.y = _msg.twist.angular.z = 0
            _msg.twist.linear.x = xd0
            #_msg.twist.linear.x = _msg.twist.linear.y = _msg.twist.linear.z = 0
            self.g_set_link_state(_msg) 
        #res = self.g_get_link_state('cart', 'beam')
        #print res
        if 1:
            q = tf.transformations.quaternion_from_euler(0., th0, 0.)
            _msg.link_name = 'pole'
            _msg.reference_frame = 'cart'
            _msg.pose.position.x, _msg.pose.position.y, _msg.pose.position.z = 0., 0., 0.
            #_msg.pose.orientation.x, _msg.pose.orientation.y, _msg.pose.orientation.z, _msg.pose.orientation.w = 0, 0, 0, 1
            _msg.pose.orientation.x, _msg.pose.orientation.y, _msg.pose.orientation.z, _msg.pose.orientation.w = q
            _msg.twist.linear.x, _msg.twist.linear.y, _msg.twist.linear.z = 0, 0, 0
            _msg.twist.angular.x, _msg.twist.angular.y, _msg.twist.angular.z = 0, 0, 0
            self.g_set_link_state(_msg)
        #res = self.g_get_link_state('pole', 'cart')
        #print res
        self.lock.acquire()
        self.position = [x0, th0]
        self.velocity = [xd0, 0]
        self.lock.release()
        #pdb.set_trace()
        self.g_unpause()
        return [x0, th0], [xd0, 0]
        
    def write_cmd_read_state(self, u):
        self.lock.acquire()
        self._u = u
        _p, _v = np.array(self.position), np.array(self.velocity)
        self.lock.release()
        return _p, _v
        
    def _run(self):
        self.pub_cmd = rospy.Publisher(self.cmd_topics, std_msgs.msg.Float64, queue_size=1)

        self.g_pause = rospy.ServiceProxy("/gazebo/pause_physics", std_srvs.srv.Empty)
        self.g_unpause = rospy.ServiceProxy("/gazebo/unpause_physics", std_srvs.srv.Empty)
        self.g_set_link_state = rospy.ServiceProxy("/gazebo/set_link_state", gazebo_msgs.srv.SetLinkState)
        self.g_get_link_state = rospy.ServiceProxy("/gazebo/get_link_state", gazebo_msgs.srv.GetLinkState)
        self.g_reset = rospy.ServiceProxy('/gazebo/reset_simulation', std_srvs.srv.Empty)

        rospy.Subscriber("/{}/joint_states".format(self.name), sensor_msgs.msg.JointState, self._on_joints_state)
        self.rate = rospy.Rate(1./self.dt)
        while not self.ros_thread_stop:
            self._publish_action()
            try:
                self.rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                pass # time goes backward when we reset gazebo

    def _publish_action(self):
        _msg = std_msgs.msg.Float64()
        self.lock.acquire()
        _msg.data = self._u
        self.lock.release()
        self.pub_cmd.publish(_msg)

    def _on_joints_state(self, msg):
        self.lock.acquire()
        self.position = msg.position
        self.velocity = msg.velocity
        self.lock.release()


        
