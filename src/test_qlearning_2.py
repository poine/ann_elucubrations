#!/usr/bin/env python
import time, numpy as np
import gym
import pdb

'''
 Qlearning on frozen lake
 This introduces a learning rate:
      https://www.practicalai.io/teaching-ai-play-simple-game-using-q-learning/

 Description of all envs: https://gym.openai.com/envs/
'''


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, observation_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class QTableAgent:
    def __init__(self, action_space, observation_space):

        self.Q = np.zeros((observation_space.n, action_space.n))
        self.action_space = action_space

    def act(self, observation, reward, done, n_action):
        pdb.set_trace()
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(self.Q[observation,:] + np.random.randn(1, self.action_space.n)*(5./(n_action+1)))
        return a
    
    

def main(env_id):
    
    env = gym.make(env_id)
    outdir = '/tmp/random-agent-results'
    env = gym.wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = QTableAgent(env.action_space, env.observation_space)
    episode_count, max_actions_per_episode, done = 5, 250, False
    reward = 0
    for i in range(episode_count):
        print('### Episode {}'.format(i))
        ob, n_actions, done = env.reset(), 0, False
        while not done and n_actions < max_actions_per_episode:
            action = agent.act(ob, reward, done, n_actions)
            n_actions += 1
            ob_1, reward, done, info = env.step(action)
            
            env.render()

    env.close()
    
if __name__ == '__main__':
    #np.set_printoptions(linewidth=300, suppress=True)
    #env = 'FrozenLake-v0'
    env = 'FrozenLake8x8-v0'
    #time.sleep(10)
    main(env)
    
