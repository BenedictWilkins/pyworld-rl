#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:59:19 2019

@author: ben
"""

import collections
import gym
import numpy as np

class ValueIteration:
    
    def __init__(self, env, gamma=0.9):  
        """
        Initialise the ValueIteration algorithm. 
        Args:
            env - training environment
            gamma - discount factor
        """
        self.env = env
        self.state = env.reset()
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
        self.gamma = gamma
     
 
    def random_step(self):
        """
            Take a random step in the training environment. Used to gather initial estimates for transition probability and rewards.
        """
        action = self.env.action_space.sample()
        new_state, reward, is_done, _ = env.step(action)
        self.rewards[(self.state, action, new_state)] = reward
        self.transitions[(self.state, action)][new_state] += 1
        self.state = self.env.reset() if is_done else new_state

    def estimate_value(self, state):
        """
            Provides an estimate of the value of the given state as: V(s) = max_a sum(p(s',r|s,a)[r + gamma V(s')])
            Args:
                state - to estimate the value of
            Returns: 
                The value of the given state, the maximal action, abs difference between the previous and new value of the state.
                (V(s), index (action) of V(s), | v - V(s) |)
        """
        values = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            target_counts = self.transitions[(state, action)]
            total = sum(target_counts.values())
            for tgt_state, count in target_counts.items():
                reward = self.rewards[(state, action, tgt_state)]
                values[action] += (count / total) * (reward + self.gamma * self.values[tgt_state])
        amax = np.argmax(values)
        v = self.values[state]
        return values[amax], amax, np.abs(v - values[amax])
        
    def greedy_policy(self, state):
        """
            The greedy policy based on current value function estimate. Has the side effect of updating the value estimate V(state).
            Args:
                state - to select greedy action for
        """
        self.values[state], action, _ = self.estimate_value(state)
        return action
        
    
    def run(self, random_steps=10000, theta=10e-5):
        """
            Runs the Value Iteration algorithm, estimates the state value function V(s) for all s
            Args:
                random_steps - how many random steps to take (see ValueIteration.random_step).
                theta - stopping condition (theta < |V(s) - V'(s)|)
        """
        # take some random steps to get initial estimate of transitions and rewards
        for _ in range(random_steps):
            self.random_step() 
        delta = np.inf
        while theta < delta:  
            delta = 0 
            for state in range(self.env.observation_space.n):
                self.values[state], _, delta2 = self.estimate_value(state)
                delta = max(delta, delta2)

    def play_episode(self, env):
        """
            Using the estimated value function V and the greedy policy, plays one episode in the given environment.
            Args: 
                env - environment to play in
            Returns:
                total reward accumulated during the episode
        """
        total_reward = 0.0
        state = env.reset()
        done = False
        while not done:
            _, action, _ = self.estimate_value(state)
            new_state, reward, done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            #env.render()
            self.transitions[(state, action)][new_state] += 1
            total_reward += reward
            state = new_state
        return total_reward
      
if __name__ == "__main__":
    import matplotlib.pyplot as plt
     
    
    def run_and_plot_values(vi):
        values = {k:[] for k in range(0,env.observation_space.n)}
        theta = 10e-5
        random_steps = 10000
        for _ in range(random_steps):
            vi.random_step() 
        delta = np.inf
        while theta < delta: 
            delta = 0 
            for state in range(vi.env.observation_space.n):
                vi.values[state], _, delta2 = vi.estimate_value(state)
                delta = max(delta, delta2)
            for s,v in vi.values.items():
                values[s].append(v)
        for s,vs in values.items():
            plt.plot(vs)
    ENV_NAME = 'FrozenLake-v0'
    env = gym.make(ENV_NAME)
    
    vi = ValueIteration(env)
    
    run_and_plot_values(vi)
    
   

        

