#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:25:51 2019

@author: ben
"""

import gym
from abc import ABC, abstractmethod

import torch.multiprocessing as mp #used for multi-environment parallelism
import collections

class Meta:
    
    def __init__(self, env=0):
        self.episode = 0
        self.step = 0
        self.global_step = 0
        self.done = 0
        self.environment = env
        
    def __str__(self):
        s = 'time(env: {4}, episode: {0}, step: {1}, global: {2}, done: {3})'.format(self.episode, self.step, self.global_step, self.done, self.environment)
        return s
    
    def __repr__(self):
        return self.__str__()

class Simulator(ABC):
    
    def __init__(self):
        self.running = False
        self.agents = []
        
    def add_agent(self, agent):
        self.agents.append(agent)
            
    def stop(self):
        self.running = False
        
    @abstractmethod
    def __iter__(self):
        pass


def environment(env):
    if isinstance(env, str):
        env = gym.make(env)
    assert isinstance(env, gym.Env)
    return env

class GymSimulator(Simulator):
    
    obs = collections.namedtuple('observation', ['action', 'reward', 'state', 'info'])
    obs_reset = collections.namedtuple('observation', ['state', 'info'])
    
    def __init__(self, envs, proc_count=12):
        super(GymSimulator, self).__init__()
        self.envs = {}
        self.meta = {}

        if isinstance(envs, str):
            self.envs[0] = environment(envs)
            self.meta[0] = Meta(0)
        elif isinstance(envs, gym.Env):
            self.envs[0] = envs
            self.meta[0] = Meta(0)
        else:
            for i in range(len(envs)):
                env = environment(envs[i])
                self.envs[i] = env
                self.meta[i] = Meta(i)
    
    def stop(self):
        self.running = False
        for _,env in self.envs.items():
            env.close()
    
    def add_agent(self, agent):
        assert len(self.agents) <= 1, 'An OpenAi-gym simulator only permits single agent environments.'
        super(GymSimulator, self).add_agent(agent)
    
    def __iter__(self):
        self.running = True
        sense_call = self.agents[0].sensor.__sense__()
        sense_call.send(None)
        reset_call = self.agents[0].sensor.__reset__()
        reset_call.send(None)
        act_call = self.agents[0].actuator.__act__()
        
        reset = [reset_call]
        sense = [sense_call]
        act = [act_call]
        
        #round robin switching between environments, this could be parallelised
        
        actions = [None] * len(self.envs)
        # get initial states for each env
        for k,env in self.envs.items():
            state = self.__reset__env__(k, reset)
            yield None, None, state, self.meta[k]
            actions[k] = next(act[0])
        
        while(self.running):
            for k, env in self.envs.items():
                action = actions[k]
                state, reward, self.meta[k].done, _ = env.step(action)
                self.meta[k].step += 1
                self.meta[k].global_step += 1
                
                for s in sense:
                    s.send(GymSimulator.obs(action, reward, state, self.meta[k]))
                
                yield action, reward, state, self.meta[k]
                
                if self.meta[k].done:
                   state = self.__reset__env__(k, reset)
                   yield None, None, state, self.meta[k]
                actions[k] = next(act[0])
               
    def __reset__env__(self, k, resets):
        self.meta[k].episode += 1
        self.meta[k].step = 0
        self.meta[k].done = False
        state = self.envs[k].reset()
        for r in resets:
            r.send(GymSimulator.obs_reset(state, self.meta[k]))
        return state









'''   
class GymSimulator(Simulator):
    
    def __init__(self, env, debug=None, render=False):
        super(GymSimulator, self).__init__()
        assert isinstance(env, gym.Env) or isinstance(env, str)
        
        if isinstance(env, gym.Env):
            self.env = env
        else:
            self.env = gym.make(env)

        self.time = Time()
        self.render = render
        self.debug = debug
    
    def stop(self):
        self.running = False
        self.env.close()
    
    def add_agent(self, agent):
        assert len(self.agents) <= 1, 'An OpenAi-gym simulator only permits single agent environments.'
        super(GymSimulator, self).add_agent(agent)
        
    def __iter__(self):
        self.running = True
        sense_call = self.agents[0].sensor.__sense__()
        sense_call.send(None)
        reset_call = self.agents[0].sensor.__reset__()
        reset_call.send(None)
        act_call = self.agents[0].actuator.__act__()
        
        reset = [reset_call]
        sense = [sense_call]
        act = [act_call]
        
        state = self.__reset__env__(reset)

        yield None, None, state, self.time
        while(self.running):
            action = next(act[0]) #combine all actions from all agents into 1?7
            state, reward, self.time.done, _ = self.env.step(action)
            if self.render:
                self.env.render()
            self.time.step += 1
            self.time.global_step += 1
            
            for s in sense:
                s.send((action, reward, state, self.time))
            
            yield action, reward, state, self.time
            
            if self.time.done:
               state = self.__reset__env__(reset)
               yield action, reward, state, self.time
               
    def __reset__env__(self, resets):
        self.time.episode += 1
        self.time.step = 0
        self.time.done = False
        state = self.env.reset()
        if self.render:
            self.env.render()
        for r in resets:
            r.send((state, self.time))
        return state
'''