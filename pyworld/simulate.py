#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:25:51 2019

@author: ben
"""

import gym
from abc import ABC, abstractmethod


class Time:
    
    def __init__(self):
        self.episode = 0
        self.step = 0
        self.global_step = 0
        self.done = 0
        
        
    def __str__(self):
        s = 'time(episode: {0} step: {1} done: {2} global: {3} )'.format(self.episode, self.step, self.done, self.global_step)
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

        yield self.time
        while(self.running):
            action = next(act[0]) #combine all actions from all agents into 1?7
            nstate, reward, self.time.done, _ = self.env.step(action)
            if self.render:
                self.env.render()
            self.time.step += 1
            self.time.global_step += 1
            
            for s in sense:
                s.send((action, reward, nstate, self.time))
                
            if self.debug:
                self.debug(self.agents[0], (state, action, reward, nstate, self.time))
                
            state = nstate
            
            yield self.time
            if(self.time.done):
               state = self.__reset__env__(reset)
               yield self.time
               
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