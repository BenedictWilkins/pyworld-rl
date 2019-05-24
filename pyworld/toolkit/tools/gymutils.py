#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:57:17 2019

@author: ben
"""
import gym 
import numpy as np

from collections import namedtuple

tuple_r = namedtuple('observation', ['reward'])
tuple_sr = namedtuple('observation', ['state', 'reward'])
tuple_sar = namedtuple('observation', ['state', 'action', 'reward'])
tuple_ars = namedtuple('observation', ['action', 'reward', 'nstate'])
tuple_sars = namedtuple('observation', ['state', 'action', 'reward', 'nstate'])


class ResetEnvWrapper(gym.Wrapper):
    
    def __init__(self, env_name, env_snapshot):
        super(ResetEnvWrapper, self).__init__(gym.make(env_name))
        self.snapshot = env_snapshot
        
    def step(self, action):
        return self.env.step(action)
        
    def reset(self, **kwargs):
        self.env.reset(**kwargs) #must be done for some reason
        self.env.unwrapped.restore_full_state(self.snapshot)
        return self.env.unwrapped._get_obs()
    
def uniform_random_policy(env):
    return lambda _: np.random.randint(0, env.action_space.n)
    
def r_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        yield tuple_r(reward)
        
def sr_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield state, reward
        state = nstate

def sar_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield tuple_sar(state, action, reward)
        state = nstate

def ars_iterator(env, policy):
    state = env.reset()
    yield None, None, state ##??
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        yield tuple_ars(action, reward, state)

def sars_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield tuple_sars(state, action, reward, nstate)
        state = nstate

def episode_iterator(iterator, env, policy, astuples=True, **kwargs):
    if astuples:
         while True:
            obs =  []
            for ob in iterator(env, policy, **kwargs):
                obs.append(ob)
            yield  obs
    else:
        while True:
            yield [ob for ob in map(np.array, zip(*iterator(env, policy, **kwargs)))]
 
def sdr_episode_iterator(env, policy, gamma=0.99, iterator=sr_iterator):
    '''
        Computes the total discounted reward for each state in a single episode
        args:
            env: environment
            policy: policy mapping states to actions
            gamma: discount factor
    '''
    for obs in episode_iterator(iterator, env, policy, astuples=False):
        #compute discounted reward in place
        rewards = obs[1]
        dr = 0
        for i in range(len(rewards)-1, -1, -1):
            rewards[i] = rewards[i] + gamma * dr
            dr = rewards[i]
        yield obs
        
'''
def dataset(iterator, file, n=1000):
    data = []
    print("INFO: ")
    for i in range(n):
        data.append(next(iterator))
        if not i % 100:
            print('item:', i)
    print('saving data')
    with open(fileutils.name_file(file), 'wb') as fp:    
        pickle.dump(data, fp)
    print('done')
'''

def atari0():
    import atari_py as ap
    games = []
    for g in ap.list_games():
        gg = g.split("_")
        gg = [g.capitalize() for g in gg]
        games.append(''.join(gg) + "-v0")
    return games


def time_slice(values):
    values = list(reversed(sorted(values, key=len)))
    rvalues = []
    for i in range(len(values[0])): 
        while len(values[-1]) <= i:
            values = values[:-1]
        rv = np.empty(len(values))
        for j in range(len(values)):
            rv[j] = values[j][i]
        rvalues.append(rv)
    return rvalues

if __name__ == "__main__":
    from tensorboardX import SummaryWriter

    EPISODES = 500
    for env_name in atari0():
        try:
            env = gym.make(env_name)
        except:
            print("no such game:", env_name)
            continue
        print("RUNNING GAME: ", env)
        policy = uniform_random_policy(env)

        iterator = sdr_episode_iterator(env, policy)
        values = []
        i = 0
        for episode in iterator:
            i += 1
            values.append(episode[1])
            if not i % EPISODES:
                break
            
        rvalues = time_slice(values)
        writer = SummaryWriter()
        for i in range(len(rvalues)):
            writer.add_histogram(env_name, rvalues[i], i)
        writer.close()     