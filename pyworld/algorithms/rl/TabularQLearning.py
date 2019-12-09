#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:22:16 2019

@author: ben
"""

import collections
import numpy as np

from pyworld.toolkit.tools.gymutils import policy

def default_dict():
    return collections.defaultdict(float)

class TabularQLearning:
  
    def __init__(self, action_space, gamma=0.99, alpha=0.1, temp=1.):
        self.values = collections.defaultdict(default_dict)
        self.action_space = action_space
        self.gamma = gamma
        self.alpha = alpha
        self.boltzman_dist = policy.value.boltzmann(temp)
    
    def update(self, s, a, r, sn):
        q = self.values[s][a]
        mq = self.values[sn][self.greedy_action(sn)] #max action or random by default
        self.values[s][a] = (1-self.alpha) * q + self.alpha * (r + self.gamma * mq)

    def action_probs(self, state):
        action_values = np.array([[k,v] for k,v in self.values[state].items()])

        if action_values.shape[0] == 0:
            return np.full(self.action_space.n, 1./self.action_space.n), np.zeros(self.action_space.n) #default uniform when we got no values yet!

        values = np.zeros(self.action_space.n)      #default Q(s,a) = 0
        values[action_values[:,0].astype(np.int8)] = action_values[:,1] #update the actions we have values for
        probs = self.boltzman_dist(values)           #get action probabilities
        return probs, values
        
    def policy(self, state):
        probs, _ = self.action_probs(state)
        return np.random.choice(np.arange(self.action_space.n), p=probs)
    
    def state_value(self, state):
        probs, values = self.action_probs(state)
        return np.dot(probs, values)
    
    def state_action_value(self, state, action):
        return self.values[state][action]
        
    def greedy_action(self, state):
        return max(self.values[state].keys(), key=(lambda action: self.values[state][action]), default=self.action_space.sample())

    


    '''
    def policy(self, state):
        if np.random.uniform() > self.epsilon:
            return self.greedy_action(state)
        else:
            return self.action_space.sample()
    '''
    
'''
class QLearning:
     
    def __init__(self, env):  
        """
        Initialise the Tabular Q-Learning algorithm. 
        Args:
            env - training environment
            gamma - discount factor
        """
        self.values = collections.defaultdict(lambda: collections.defaultdict(float))
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.random_action = (lambda: random.randint(0, env.action_space.n-1))
        self.greedy_action = (lambda s: max(self.values[s].keys(), key=(lambda k: self.values[s][k]), default=self.random_action()))

    def update_value(self, s, a, r, sn, alpha = 0.1, gamma = 0.9):
        q = self.values[s][a]
        mq = self.values[sn][self.greedy_action(sn)] #max action or random by default
        self.values[s][a] = (1-alpha) * q + alpha * (r + gamma * mq)
    
    def e_greedy_policy(self, s, epsilon=0.1):
        r = np.random.uniform()
        if(r > epsilon):
            return self.greedy_action(s)
        else:
            return self.random_action()
    
    def __play_episode_render(self, env, policy):
        total_reward = 0.0
        s = env.reset()
        done = False
        while not done:
            a = policy(s)
            sn, r, done, _ = env.step(a)
            total_reward += r
            self.update_value(s,a,r,sn)
            s = sn
            env.render()
        return total_reward
        
    def __play_episode(self, env, policy):
        total_reward = 0.0
        s = env.reset()
        done = False
        while not done:
            a = policy(s)
            #print(a)
            sn, r, done, _ = env.step(a)
            total_reward += r
            self.update_value(s,a,r,sn)
            s = sn
        return total_reward

    def play_episode(self, env, policy=None, render=False):
        """
            Runs the Value Iteration algorithm, estimates the state value function V(s) for all s
            Args:
                random_steps - how many random steps to take (see ValueIteration.random_step).
                theta - stopping condition (theta < |V(s) - V'(s)|)
        """
        if not policy:
            policy = self.e_greedy_policy
        if(render):
            return self.__play_episode_render(env, policy)
        else:
            return self.__play_episode(env, policy)
        
    def to_heat_map(self, grid, i):
        m = np.zeros((self.observation_space.n, self.action_space.n))
        for s,aa in self.values.items():
            for a,v in aa.items():
                m[s,a] = v
        return m
    
        
        
        
   
if __name__ == "__main__":   
    ENV_NAME = 'FrozenLake-v0'
    env = gym.make(ENV_NAME)
    ql = QLearning(env)
    ANIMATE_EVERY = 50
    hmap = Ani.HeatmapAnimation(np.zeros((env.observation_space.n, env.action_space.n)), ql.to_heat_map)
    hmap.interactive()
    i = 0
    epsilon = 1.
    while not hmap.stop_interactive:
        i += 1
        epsilon = 1000./i
        ql.play_episode(env, lambda s: ql.e_greedy_policy(s, epsilon=epsilon), render=False)
        #print("Episode:",i,"Total Reward:", ql.play_episode(env, render=False))
        if i % ANIMATE_EVERY == 0:
            hmap.interactive_update(i)
    total = 0
    TEST_EPISODES = 100
    for i in range(TEST_EPISODES):
        #ql.play_episode(env, lambda s: ql.e_greedy_policy(s, epsilon=0.0), render=False)
        total += ql.play_episode(env, lambda s: ql.e_greedy_policy(s, epsilon=0.0), render=False)
    print("TotalReward:",total/TEST_EPISODES)
'''