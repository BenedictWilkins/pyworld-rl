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
import itertools

if __name__ != "__main__":
    from . import datautils as du

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
        
    
state = 'state'
action = 'action'
reward = 'reward'
next_state = 'nstate'   

s = namedtuple('observation', ['state'])
s.__new__.__defaults__ = (None,)
r = namedtuple('observation', ['reward'])
r.__new__.__defaults__ = (None,)
sa = namedtuple('observation', ['state', 'action'])
sa.__new__.__defaults__ = (None,None)
ss = namedtuple('observation', ['state', 'nstate'])
ss.__new__.__defaults__ = (None,None)
sr = namedtuple('observation', ['state', 'reward'])
sr.__new__.__defaults__ = (None,None)
sar = namedtuple('observation', ['state', 'action', 'reward'])
sar.__new__.__defaults__ = (None,None,None)
ars = namedtuple('observation', ['action', 'reward', 'nstate'])
ars.__new__.__defaults__ = (None,None,None)
sas = namedtuple('observation', ['state', 'action', 'nstate'])
sas.__new__.__defaults__ = (None,None,None)
sars = namedtuple('observation', ['state', 'action', 'reward', 'nstate'])
sars.__new__.__defaults__ = (None,None,None,None)



def s_iterator(env, policy):
        state = env.reset()
        yield s(state), False
        done = False
        while not done:
            action = policy(state)
            state, _, done, _ = env.step(action)
            yield s(state), done
        
def r_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        yield r(reward), done
        
def sa_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, _ = env.step(action)
        yield sa(state, action), done
        state = nstate
        
def sr_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield sr(state, reward), done
        state = nstate
        
        
def ss_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, _ = env.step(action)
        yield ss(state, nstate), done
        state = nstate

def sar_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield sar(state, action, reward), done
        state = nstate

def ars_iterator(env, policy):
    state = env.reset()
    yield ars(None, None, state), False ##??
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        yield ars(action, reward, state), done

def sas_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, _ = env.step(action)
        yield sas(state, action, nstate), done
        state = nstate

def sars_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield sars(state, action, reward, nstate), done
        state = nstate
        
iterators = {s:s_iterator, r:r_iterator, sa:sa_iterator, ss:ss_iterator, sr:sr_iterator, 
             sar:sar_iterator, ars:ars_iterator, sas:sas_iterator, 
             sars:sars_iterator}
    
class GymIterator:

    def __init__(self, _env, _policy, mode = s, episodic=True):
        self.episodic = episodic
        self.__done = False
        self.__env = _env
        self.__policy = _policy
        self.__iterator_type = iterators[mode]
        self.__iterator = self.__iterator_type(self.__env, self.__policy)
     
    def reset(self):
        self.__done = False
        self.__iterator = self.__iterator_type(self.__env, self.__policy)
        
    def __getitem__(self, _slice):
        return itertools.islice(self, _slice.start, _slice.stop, _slice.step)
    
    def __next__(self):
        if self.__done:
            self.reset()
            if self.episodic:
                raise StopIteration
        result, self.__done = next(self.__iterator)
        return result
    
    def __iter__(self):
        return self

def dynamic_dataset(env, policy, mode = s, chunk = 1, size = 1000, random = True):
    iterator = GymIterator(env,policy, mode, episodic=False)
    return du.dynamic_dataset(iterator, chunk, size, random)
        
def dataset(env, policy, mode = s, size = 1000):
    iterator = GymIterator(env,policy, mode, episodic=False)
    return du.dataset(iterator, size)
    
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

def assert_box(space):
    assert isinstance(space, gym.spaces.Box)
    
def assert_interval(space, high=255, low=0):
    u = np.unique(space.high.ravel())
    assert len(u) == 1
    assert u[0] == high
    u = np.unique(space.low.ravel())
    assert len(u) == 1
    assert u[0] == low
    
def assert_unique(space):
    high = np.unique(space.high.ravel())
    assert len(high) == 1
    low = np.unique(space.low.ravel())
    assert len(low) == 1
    return high[0], low[0]
    
    
class __OM_Gray:
    
    def __init__(self, env):
        assert_box(env.observation_space)
        assert_interval(env.observation_space)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(*env.observation_space.shape[:2], 1), dtype=np.float32)
    
    def __call__(self, state, *args):
        state = (state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114).astype(np.float) / 255.
        return state.reshape(*state.shape, 1)
    
class __OM_Interval:
    
    def __init__(self, env):
        assert_box(env.observation_space)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)
    
    def __call__(self, state, *args):
        state.astype(np.float) / 255.0
        
class __OM_Crop:
    
    def __init__(self, env, shape):
        assert_box(env.observation_space)
        self.observation_space = env.observation_space #gym.spaces.Box(low=env.observation_space.low, high=env.observation_space.high, shape=shape, dtype=np.float32)
    
    def __call__(self, state, shape):
        raise NotImplementedError()
    
class __OM_Resize:
    
    def __init__(self, env, shape):
        assert_box(env.observation_space)
        high, low = assert_unique(env.observation_space)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    
    def __call__(self, state):
        return cv2.resize(state, self.observation_space.shape, interpolation=cv2.INTER_AREA)


class __OM_Binary:
    def __init__(self, env, threshold):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        
        self.threshold = threshold * (high - low)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)
    
    def __call__(self, state):
        return np.where(state > self.threshold, 1., 0.)
        
class __OM_Default:
    
    def __init__(self, env):
        assert_box(env.observation_space)
        assert_interval(env.observation_space)
        self.observation_space = gym.spaces.Box(low=0., high=1.0, shape=(84, 84, 1), dtype=np.float32)
        
    def __call__(self, state):
        img = (state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114).astype(np.float32) #gray
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA) #resize
        img = img[18:102, :] / 255.0 #crop
        return img.reshape((*img.shape, 1))  

class __OM_CHW:
    
    def __init__(self, env):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        channels = env.observation_space.shape[2]
        assert channels == 1 or channels == 3 or channels == 4
        self.observation_space = gym.spaces.Box(low = low, high = high, shape=(channels, *env.observation_space.shape[0:2]), dtype = env.observation_space.dtype)
        
    def __call__(self, state):
        return np.swapaxes(state, 0, 2)


observation_mode = namedtuple('observation_transform', 'gray interval crop resize binary chw default')(
                                    __OM_Gray, __OM_Interval, __OM_Crop, __OM_Resize, __OM_Binary, __OM_CHW, __OM_Default)

class ObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env, mode, *modeargs):
        super(ObservationWrapper, self).__init__(env)
        self.mode = mode(env, *modeargs)
        self.observation_space = self.mode.observation_space
        
    def observation(self, obs):
        return self.mode(obs)

    
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
    import gym
    import pyworld.toolkit.tools.visutils as vu
    import pyworld.toolkit.tools.datautils as du
    
    env = gym.make('Pong-v0')
    env = ObservationWrapper(env, observation_mode.default)
    env = ObservationWrapper(env, observation_mode.chw)
    print(env.observation_space)
    
    def video(env, policy):
        for state in GymIterator(env, policy, mode=s):
            yield state.state
    
    #vu.play(video(env, uniform_random_policy(env)))
    
    
    
    
    
    
    
    
    