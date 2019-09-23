#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:57:17 2019

@author: ben
"""
import gym 
import numpy as np
import cv2

from collections import namedtuple
from . import datautils as du

tuple_s = namedtuple('observation', ['state'])
tuple_r = namedtuple('observation', ['reward'])
tuple_sa = namedtuple('observation', ['state', 'action'])
tuple_sr = namedtuple('observation', ['state', 'reward'])
tuple_sar = namedtuple('observation', ['state', 'action', 'reward'])
tuple_ars = namedtuple('observation', ['action', 'reward', 'nstate'])
tuple_sas = namedtuple('observation', ['state', 'action', 'nstate'])
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
    
def uniform_random_policy(env, onehot=False):
    if onehot:
        return lambda _: du.onehot_int(env.action_space.sample(), env.action_space.n)
    else:
        return lambda _: env.action_space.sample()

def vis_iterator(iterator):
    for x in iterator:
        yield x.state

def s_iterator(env, policy):
    state = env.reset()
    yield tuple_s(state)
    done = False
    while not done:
        action = policy(state)
        state, _, done, _ = env.step(action)
        yield tuple_s(state)
    
def r_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        yield tuple_r(reward)
        
def sa_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, _ = env.step(action)
        yield tuple_sa(state, action)
        state = nstate
        
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

def sas_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, _ = env.step(action)
        yield tuple_sars(state, action, nstate)
        state = nstate

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


    
class ObservationWrapper(gym.ObservationWrapper):
      
    GRAY_SCALE_CROP = lambda _: ((1,84,84), ObservationWrapper.__gray_scale_crop)
    GRAY = lambda env: ((1, *env.observation_space.shape[:2]), ObservationWrapper.__gray)
    IDENTITY = lambda env: ((env.observation_space.shape[2], *env.observation_space.shape[:2]), ObservationWrapper.__identity)
    
    def __init__(self, env, processor):
        super(ObservationWrapper, self).__init__(env)
        self.shape, self.__c = processor(env)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.shape) 
    
    def observation(self, obs):
        return self.__c(obs, self.shape)

    def __identity(state, shape):
        return np.reshape(state, shape).astype(np.float) / 255.0
        
    def __gray(state, shape):
        img = (state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114).astype(np.float)
        return np.reshape(img, shape).astype(np.float) / 255.0
        
    def __gray_scale_crop(state, shape):
        img = (state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114).astype(np.float)
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        return np.reshape(img[18:102, :], shape).astype(np.float) / 255.0
    
    def __binarise(state, threshold=0.5):
        return np.where(state>threshold, 1., 0.)
    
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