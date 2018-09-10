import os, shutil, logging, yaml, argparse, pickle, pprint, numpy as np
import tensorflow as tf, tflearn
import gym, gym_foo
import ddpg_utils
import pdb

LOG = logging.getLogger('ddpg')
# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


class Agent:
    def __init__(self, sess, env, config):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high
        print('in Agent::__init__: state_dim {} action_dim {}'.format(self.state_dim, self.action_dim))
        # Ensure action bound is symmetric
        assert (np.all(env.action_space.high == -env.action_space.low))

        self.actor = ActorNetwork(sess, self.state_dim, self.action_dim, self.action_bound,
                                  config['agent']['actor']['learning_rate'],
                                  config['agent']['tau'],
                                  config['agent']['actor']['minibatch_size'])

        self.critic = CriticNetwork(sess, self.state_dim, self.action_dim,
                                    config['agent']['critic']['learning_rate'],
                                    config['agent']['tau'],
                                    config['agent']['critic']['gamma'],
                                    self.actor.get_num_trainable_vars())
        
        self.actor_noise = ddpg_utils.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim),
                                                                   sigma=config['agent']['actor']['noise_sigma'])

        self.replay_buffer = ddpg_utils.ReplayBuffer(config['agent']['buffer_size'], config['random_seed'])

        self.init_training(sess, env, config)

    # ===========================
    #   Tensorflow Summary Ops
    # ===========================

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar("Qmax_Value", episode_ave_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar("Duration", episode_duration)
        
        self.summary_vars = [episode_reward, episode_ave_max_q, episode_duration]
        self.summary_ops = tf.summary.merge_all()

        return self.summary_ops, self.summary_vars

    # ===========================
    #   Agent Training
    # ===========================
    def init_training(self, sess, env, args):
        
        # Set up summary Ops
        summary_ops, summary_vars = self.build_summaries()

        sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        # Needed to enable BatchNorm. 
        # This hurts the performance on Pendulum but could be useful
        # in other environments.
        if args['enable_batch_norm']:
            tflearn.is_training(True)
        self.episode_nb = 0

    
    
    def train(self, sess, env, cfg):

        while self.episode_nb < cfg['max_episodes']:
            self.run_training_episode(sess, env, cfg)

            
    def run_training_episode(self, sess, env, cfg):
            s = env.reset()
            self.actor_noise.set_sigma(cfg['agent']['actor']['noise_sigma'])
            ep_reward = 0
            ep_ave_max_q = 0

            for j in range(int(cfg['max_episode_len'])):

                if cfg['render_env']:
                    env.render()

                # Compute action
                a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) + self.actor_noise()
                #a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) + np.exp(-0.003*i)*self.actor_noise()

                s2, r, terminal, info = env.step(a[0])

                self.replay_buffer.add(np.reshape(s, (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), r,
                                       terminal, np.reshape(s2, (self.actor.s_dim,)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if self.replay_buffer.size() > cfg['agent']['actor']['minibatch_size']:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        self.replay_buffer.sample_batch(cfg['agent']['actor']['minibatch_size'])

                    # Calculate targets
                    target_q = self.critic.predict_target(
                        s2_batch, self.actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(cfg['agent']['actor']['minibatch_size']):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = self.critic.train(
                        s_batch, a_batch, np.reshape(y_i, (cfg['agent']['actor']['minibatch_size'], 1)))

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(s_batch)
                    grads = self.critic.action_gradients(s_batch, a_outs)
                    self.actor.train(s_batch, grads[0])

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                s = s2
                ep_reward += r

                if terminal or j >= int(cfg['max_episode_len'])-1:
                    if j > 0: ep_ave_max_q /= float(j)
                    self.report_episode(sess, j, ep_reward, ep_ave_max_q)
                    break
            self.episode_nb += 1

    def report_episode(self, sess, ep_len, ep_reward, ep_ave_max_q):
        summary_str = sess.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: ep_reward,
            self.summary_vars[1]: ep_ave_max_q,
            self.summary_vars[2]: float(ep_len)
        })
        self.writer.add_summary(summary_str, self.episode_nb)
        self.writer.flush()

        fmt = '| Episode: {: 4d} | Reward: {: 4d} | Qmax: {:.4f} | len {: 4d} | rep buf {}'
        print(fmt.format(self.episode_nb, int(ep_reward), ep_ave_max_q , ep_len, self.replay_buffer.size()))

                
    def test(self, sess, env, args):
        s = env.reset()
        ep_reward = 0.
        for j in range(int(args['max_episode_len'])):
            if args['render_env']:
                env.render()
            a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim)))
            s, r, terminal, info = env.step(a[0])
            ep_reward += r
            if terminal or j >= int(args['max_episode_len'])-1:
                print('test done: reward {}'.format(ep_reward))
                break

    def save(self, sess, dirname):
        LOG.info('  Saving agent to directory {}'.format(dirname))
        if os.path.isdir(dirname):
            print('Model save directory exist ({}): Deleting it before proceeding'.format(dirname))
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        
        #inputs, outputs = {'act_in':self.actor.inputs}, {'act_scaled_out':self.actor.scaled_out} 
        #tf.saved_model.simple_save(sess, dirname, inputs, outputs)
        saver = tf.train.Saver()
        #saver.save(sess,  dirname+'my-model')
        saver.save(sess,  os.path.join(dirname, 'my-model'))
        with open(os.path.join(dirname, 'replay_buf'), "wb") as f:
            pickle.dump([self.replay_buffer], f)
        with open(os.path.join(dirname, 'state'), "wb") as f:
            pickle.dump([self.episode_nb], f)
            
    def load(self, sess, export_dir):
        LOG.info('  Loading agent from directory {}'.format(export_dir))
        sess.run(tf.global_variables_initializer())
        #tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(export_dir,'my-model'))
        with open(os.path.join(export_dir, 'replay_buf'), "rb") as f:
            (self.replay_buffer, ) = pickle.load(f)
        with open(os.path.join(export_dir, 'state'), "rb") as f:
            (self.episode_nb, ) = pickle.load(f)


class Config:
    def __init__(self):
        pass

    def setup_cmd_line_parser(self):
        parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
        # agent parameters
        parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
        parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
        parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
        parser.add_argument('--tau', help='soft target update parameter', default=0.001)
        parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
        parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)
        parser.add_argument('--enable-batch-norm', help='enable batch normalization',  action='store_true', default=False)
        parser.add_argument('--actor-noise-sigma', help='exploration noise', action='store')#default=0.3)
        # run parameters
        parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', action='store')
        parser.add_argument('--random-seed', help='random seed for repeatability', action='store')
        parser.add_argument('--max-episodes', help='max num of episodes to do while training', action='store')#default=50000)
        parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
        parser.add_argument('--render-env', help='render the gym env', action='store_true')
        parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
        parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
        parser.add_argument('--summary-dir', help='directory for storing tensorboard info', action='store')#default='./results/tf_ddpg')

        parser.add_argument('--config', help='read_configuration from file', default=None)
        return parser

    def parse_cmd_line(self, parser):
        args = vars(parser.parse_args())
        #print('dumping command line')
        #pprint.pprint(args)

        if args['config'] is not None: self.load(args['config'])
        
        if args['random_seed'] is not None: self.cfg['random_seed'] = int(args['random_seed'])
        if args['env'] is not None: self.cfg['env']['name'] = args['env']
        if args['render_env'] is not None: self.cfg['render_env'] = True

        if args['max_episodes'] is not None: self.cfg['max_episodes'] = int(args['max_episodes'])
        if args['summary_dir'] is not None: self.cfg['summary_dir'] = args['summary_dir']
        if args['actor_noise_sigma'] is not None: self.cfg['agent']['actor']['noise_sigma'] = float(args['actor_noise_sigma'])
        
        return args
        
    def load(self, filename):
        with open( filename, 'r') as stream:
            self.cfg = yaml.load(stream)

    def dump(self):
        print('dumping config')
        pprint.pprint(self.cfg)

    def set(self, keys, val):
        d = self.cfg
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = val

    def get(self, keys):
        d = self.cfg
        for key in keys:
            d = d[key]
        return d


class Model:
    def __init__(self, config):
        self.config = config
        self.session = tf.Session()
        self.session.__enter__()
        self.env = gym.make(self.config.cfg['env']['name'])
        try:  self.env.load_config(self.config.cfg['env'])
        except AttributeError:
            pass
        np.random.seed(self.config.cfg['random_seed'])
        tf.set_random_seed(self.config.cfg['random_seed'])
        self.env.seed(self.config.cfg['random_seed'])
        
        self.agent = Agent(self.session, self.env, self.config.cfg)
        self.is_training = False

    def __enter__(self):
        print('in Model::enter')
        return self
        
    def __exit__(self, type, value, traceback):
        print('in Model::exit')
        self.session.__exit__(type, value, traceback)
        
    def load_agent(self, export_dir):
        self.agent.load(self.session, export_dir)

    def save_agent(self, export_dir):
        self.agent.save(self.session, export_dir)
        
    def test_agent(self):
        self.agent.test(self.session, self.env, self.config.cfg)

    def init_training(self):
        if os.path.isdir(self.config.cfg['summary_dir']):
            print('Model summary directory exist ({}): Deleting it before proceeding'.format(self.config.cfg['summary_dir']))
            shutil.rmtree(self.config.cfg['summary_dir'])
        self.agent.init_training(self.session, self.env, self.config.cfg)

    def run_training(self):
        self.is_training = True
        self.stop_requested = False
        while self.agent.episode_nb < self.config.cfg['max_episodes'] and not self.stop_requested:
            self.agent.run_training_episode(self.session, self.env, self.config.cfg)
        #self.agent.train(self.session, self.env, self.config.cfg)
        self.is_training = False
        
    def abort_training(self):
        self.stop_requested = True
        
    def train_agent(self):
        #self.init_training()
        self.run_training()
