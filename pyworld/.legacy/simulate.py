#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:25:51 2019

@author: ben
"""

import gym
import math
from abc import ABC, abstractmethod

import torch.multiprocessing as mp #used for multi-environment parallelism
import collections

class Meta:
    
    def __init__(self, name=0):
        self.episode = 0
        self.step = 0
        self.global_step = 0
        self.done = 0
        self.name = name
        
    def __str__(self):
        s = 'info(name: {4}, episode: {0}, step: {1}, global: {2}, done: {3})'.format(self.episode, self.step, self.global_step, self.done, self.name)
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
    def __call__(self):
        pass

def environment(env):
    if isinstance(env, str):
        env = gym.make(env)
    assert isinstance(env, gym.Env)
    return env

def iterable(obj):
    try:
        iter(obj)
    except TypeError:
        obj = [obj]
    return obj

def chunk(arg, n):
    for i in range(0,len(arg), n):
        yield arg[i:i + n]

class GymSimulator(Simulator):
    
    obs_tuple = collections.namedtuple('observation', ['action', 'reward', 'state', 'info'])
    obs_reset_tuple = collections.namedtuple('observation', ['state', 'info'])
    callbacks_tuple = collections.namedtuple('callbacks', ['sense', 'reset', 'act'])
    env_tuple = collections.namedtuple('environment', ['env', 'meta', 'callbacks'])
    
    def __init__(self, envs):
        super(GymSimulator, self).__init__()
        self.envs = []
        self._init_envs = iterable(envs)

        
    def stop(self):
        self.running = False
        for env in self.envs:
            env.env.close()
    
    def add_agent(self, agent):
        assert len(self.agents) <= 1, 'An OpenAi-gym simulator only permits single agent environments.'
        super(GymSimulator, self).add_agent(agent)
    
    def __call__(self, procs=None):
        self.running = True
        if procs is not None:
            split = max(1, math.ceil(len(self._init_envs)/ procs))
            env_procs = [l for l in chunk(self._init_envs, split)]
            self.__init_and_run_envs_mp(env_procs)
        else:
            for meta in self.__init_and_run_envs(self._init_envs):
                yield meta

            
    def __init_envs(self, init_envs, agent):
        envs = []
        for i in range(len(init_envs)):
            env = environment(init_envs[i])
            env = GymSimulator.env_tuple(env, Meta(i), GymSimulator.callbacks_tuple([], [], []))
            #mp will take deep copies of sensors and actuators automatically.
            env.callbacks.sense.append(agent.sensor._sense_callback)
            env.callbacks.reset.append(agent.sensor._reset_callback)
            env.callbacks.act.append(agent.actuator.__act__())
            envs.append(env)
        return envs
        
    def __init_and_run_envs_mp(self, envs_procs):
        processes = []
        
        for envs in envs_procs:
            p = mp.Process(target=self.__init_and_run_envs, args=(envs,)) #TODO
            p.start()
            processes.append(p)
        for p in processes:
            p.join()   

    def __init_and_run_envs(self, envs):
        #initialise all environments to run, Gym environments only support 1 agent
        print(self, "running environments: ", envs)
        self.envs = envs = self.__init_envs(envs, self.agents[0])
        #round robin switching between environments
        actions = [0] * len(envs)
        # get initial states for each env
        for i in range(len(envs)):
            env = envs[i]
            state = self.__reset_env(env)
            #yield None, None, state, env.meta
            for acts in env.callbacks.act:
                #this should only loop once for a gym environment
                actions[i] = next(acts) 
        yield [env.meta for env in envs]
        while(self.running):
            for i in range(len(envs)):
                env = envs[i]
                state, reward, env.meta.done, _ = env.env.step(actions[i])
                env.meta.step += 1
                env.meta.global_step += 1
                for s in env.callbacks.sense:
                    s.send(GymSimulator.obs_tuple(actions[i], reward, state, env.meta))

                if env.meta.done:
                   state = self.__reset_env(env)

                for acts in env.callbacks.act:
                    #this should only loop once for a gym environment
                    actions[i] = next(acts) 
            yield [env.meta for env in envs]
               
    def __reset_env(self, env):
        env.meta.episode += 1
        env.meta.step = 0
        env.meta.done = False
        state = env.env.reset()
        for s in env.callbacks.reset:
            s.send(GymSimulator.obs_reset_tuple(state, env.meta))
        return state


if __name__ == "__main__":
     
    import toolkit.tools.gymutils as gu
    import agent as pwag
   
    env_name = "Pong-v0"
    env = gym.make(env_name)
    
    #env3 = gym.make("Pong-v0")

    policy = gu.uniform_random_policy(env)
    sensor = pwag.MaxPoolSensor(pwag.AtariImageSensor())
    actuator = pwag.RandomActuator(env.action_space)
    ag = pwag.SimpleAgent(sensor, actuator)
    
    sim = GymSimulator([env_name,env_name])#, env2, env3])
    sim.add_agent(ag)
    
    sim(procs=2)
    
    
    