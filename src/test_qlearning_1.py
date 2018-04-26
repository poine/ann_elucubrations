#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" 

   A simple Q-Learning example
     see: http://mnemstudio.org/path-finding-q-learning-tutorial.htm

 
"""

import numpy as np
import pdb


'''
  Purely random training (exploration)
'''
def train(R, nb_episodes, gamma, goal_state, nb_state):
    Q = np.zeros((nb_state, nb_state))
    for episode in range(nb_episodes):
        state = state0 = np.random.randint(0, nb_state)
        #print state0
        while state != goal_state:
            potential_next_states = np.argwhere(R[state] >= 0).reshape((-1,))
            next_state = potential_next_states[np.random.randint(0, len(potential_next_states))]
            #print state, next_state
            #pdb.set_trace()
            Q[state, next_state] = R[state, next_state] + gamma * np.max(Q[next_state, :])
            state = next_state
    print Q
    #print 'done'
    return Q


'''
  Purely exprience based behaviour
'''
def solve(Q, start_state, goal_state):
    state = start_state
    while state != goal_state:
        print state
        state = np.argmax(Q[state, :])
    print goal_state


'''
  
'''
def train_and_solve(R, nb_episodes, gamma, exploration_rate, goal_state, nb_state):
    Q = np.zeros((nb_state, nb_state))
    for episode in range(nb_episodes):
        print('## episode {}'.format(episode))
        state = state0 = np.random.randint(0, nb_state)
        episode_path = [state0]
        while state != goal_state:
            explore = np.random.uniform(low=0.0, high=1.0, size=1) < exploration_rate
            no_map = not np.any(Q[state, :])
            print('state {} explore: {} no_map:{}'.format(state, explore, no_map))
            if explore or no_map:
                potential_next_states = np.argwhere(R[state] >= 0).reshape((-1,))
                next_state = potential_next_states[np.random.randint(0, len(potential_next_states))]
                Q[state, next_state] = R[state, next_state] + gamma * np.max(Q[next_state, :])
            else:
                next_state = np.argmax(Q[state, :])
            state = next_state
            episode_path.append(state)
                        
        print('duration {} ({})'.format(len(episode_path), episode_path))
        print('Q\n{}'.format(Q))



        
def main():

    R = np.array([[-1, -1, -1,  -1, 0,  -1],
                  [-1, -1, -1,  0, -1, 100],
                  [-1, -1, -1,  0, -1,  -1],
                  [-1,  0,  0, -1,  0,  -1],
                  [ 0, -1, -1,  0, -1, 100],
                  [-1,  0, -1, -1,  0, 100]])

    nb_state, goal_state = 6, 5
    if 0:
        Q = train(R, nb_episodes=100, gamma=0.05, goal_state=goal_state, nb_state=nb_state)
        solve(Q, 2, goal_state)
    else:
        train_and_solve(R, nb_episodes=100, gamma=0.05, exploration_rate=0.1, goal_state=goal_state, nb_state=nb_state)
        
if __name__ == '__main__':
    np.set_printoptions(linewidth=300, suppress=True)
    main()
    
