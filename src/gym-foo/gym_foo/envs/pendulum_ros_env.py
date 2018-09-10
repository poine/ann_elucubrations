'''

   openai gym ros pendulum

'''
import gym, gym.utils.seeding, math, numpy as np
import os, threading, time
import rospy, std_msgs.msg, sensor_msgs.msg, gazebo_msgs.msg, std_srvs.srv, gazebo_msgs.srv
import tf.transformations, pdb

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class PendulumRosEnv(gym.Env):

    def __init__(self):
        self._dt = 0.05
        self.max_speed, self.max_torque = 8., 5.
        self.action_space = gym.spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        high = np.array([1., 1., self.max_speed])
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
    
        self._step = 0
        self.seed()
        
        self.ros_thread = threading.Thread(target=self._run)
        self.ros_thread_stop = False
        self.lock = threading.Lock()
        self._angle, self._rvel, self._u = [0, 0], [0, 0], 0
        rospy.init_node('pendulum_ros_ctl', disable_signals=True)
        self.ros_thread.start()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def shutdown(self):
        rospy.signal_shutdown('fff')
    
    def __talk_with_ros(self, u):
        self.lock.acquire()
        _a, _v = self._angle[0], self._rvel[0]
        self._u = u
        self.lock.release()
        return _a, _v

    def __get_obs(self, _a, _v):
        return np.array([np.cos(_a), np.sin(_a), _v])
    
    def step(self,u):
        _now = time.time()
        _dt = _now - self._last_step_time
        if _dt < self._dt: time.sleep(self._dt-_dt)
        self._last_step_time = _now
        #print('{} {}'.format(self._step, _dt))
        self._step+=1
        _a, _v = self.__talk_with_ros(u)
        _cost = angle_normalize(_a)**2 + .1*_v**2 + .01*(u[0]**2)
        #pdb.set_trace()
        #print _cost
        return self.__get_obs(_a, _v), -_cost, False, {}

    def reset(self):
        _a, _v = self.np_random.uniform(low=[-np.pi, -1], high=[np.pi, 1])
        self._publish_state(_a, _v)
        self._step = 0
        self._last_step_time = time.time()
        _a, _v = self.__talk_with_ros(0)
        return self.__get_obs(_a, _v)


    def _run(self):
        self.pub_cmd = rospy.Publisher('/double_pendulum/joint1_effort_controller/command', std_msgs.msg.Float64, queue_size=1)
        self.g_pause = rospy.ServiceProxy("/gazebo/pause_physics", std_srvs.srv.Empty)
        self.g_unpause = rospy.ServiceProxy("/gazebo/unpause_physics", std_srvs.srv.Empty)
        self.g_set_link_state = rospy.ServiceProxy("/gazebo/set_link_state", gazebo_msgs.srv.SetLinkState)
        rospy.Subscriber("/double_pendulum/joint_states", sensor_msgs.msg.JointState, self._on_joints_state)
        self.rate = rospy.Rate(1./self._dt)
        while not self.ros_thread_stop:
            self._publish_action()
            self.rate.sleep()

    def _publish_action(self):
        _msg = std_msgs.msg.Float64()
        self.lock.acquire()
        _msg.data = self._u
        self.lock.release()
        #_msg.data = np.sin(rospy.Time.now().to_sec())
        self.pub_cmd.publish(_msg)

    def _publish_state(self, _a, _v):
        self.g_pause()
        q = tf.transformations.quaternion_from_euler(0., _a, 0.)
        #print a, r, q
        _msg = gazebo_msgs.msg.LinkState()
        _msg.link_name = 'link2'
        _msg.reference_frame = 'link1'#world'
        _msg.pose.position.x, _msg.pose.position.y, _msg.pose.position.z = 0, 0.1, 1.95 
        _msg.pose.orientation.y,_msg.pose.orientation.w = q[1], q[3]
        #_msg.pose.orientation.y, _msg.pose.orientation.w= -1, 0
        #_msg.twist.angular.x = _msg.twist.angular.y = _msg.twist.angular.z = 0
        _msg.twist.angular.y = _v
        _msg.twist.linear.x = _msg.twist.linear.y = _msg.twist.linear.z = 0
        self.g_set_link_state(_msg)
        self.g_unpause()
        
    def _on_joints_state(self, msg):
        self._angle = msg.position
        self._rvel = msg.velocity

class DoublePendulumRosEnv(PendulumRosEnv):
    def __init__(self):
        PendulumRosEnv.__init__(self)
        high = np.array([1., 1., self.max_speed, 1, 1, self.max_speed])
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        

    def __talk_with_ros(self, u):
         self.lock.acquire()
         _a1, _v1, _a2, _v2 = self._angle[0], self._rvel[0], self._angle[1], self._rvel[1]
         self._u = u
         self.lock.release()
         return _a1, _v1, _a2, _v2

    
    def __get_obs(self, _a1, _v1, _a2, _v2):
        return np.array([np.cos(_a1), np.sin(_a1), _v1, np.cos(_a2), np.sin(_a2), _v2])
    
    def step(self,u):
        _now = time.time()
        _dt = _now - self._last_step_time
        if _dt < self._dt: time.sleep(self._dt-_dt)
        self._last_step_time = _now
        self._step+=1
        _a1, _v1, _a2, _v2 = self.__talk_with_ros(u)
        _cost = 0.5*angle_normalize(_a1)**2 + .05*_v1**2 + 1.*angle_normalize(_a2)**2 + 0.1*_v2**2 + .01*(u[0]**2)
        return self.__get_obs(_a1, _v1, _a2, _v2), -_cost, False, {}

    def reset(self):
        _a, _v = self.np_random.uniform(low=[-np.pi, -1], high=[np.pi, 1])
        self._publish_state(_a, _v)
        self._step = 0
        self._last_step_time = time.time()
        _a1, _v1, _a2, _v2 = self.__talk_with_ros(0)
        return self.__get_obs(_a1, _v1, _a2, _v2)

    def _publish_state(self, _a, _v):
        self.g_pause()
        q = tf.transformations.quaternion_from_euler(0., _a, 0.)
        #print a, r, q
        _msg = gazebo_msgs.msg.LinkState()
        _msg.link_name = 'link2'
        _msg.reference_frame = 'link1'
        _msg.pose.position.x, _msg.pose.position.y, _msg.pose.position.z = 0, 0.1, 1.95 
        _msg.pose.orientation.y,_msg.pose.orientation.w = q[1], q[3]
        #_msg.pose.orientation.y, _msg.pose.orientation.w= -1, 0
        #_msg.twist.angular.x = _msg.twist.angular.y = _msg.twist.angular.z = 0
        _msg.twist.angular.y = _v
        _msg.twist.linear.x = _msg.twist.linear.y = _msg.twist.linear.z = 0
        self.g_set_link_state(_msg)

        _msg.link_name = 'link3'
        _msg.reference_frame = 'link2'
        _msg.pose.position.x, _msg.pose.position.y, _msg.pose.position.z = -0.597857129633, 0.199980542063, 1.277265459
        _msg.pose.orientation.x, _msg.pose.orientation.y, _msg.pose.orientation.z, _msg.pose.orientation.w= 0, 0, 0, 1
        
        self.g_set_link_state(_msg)
        self.g_unpause()
