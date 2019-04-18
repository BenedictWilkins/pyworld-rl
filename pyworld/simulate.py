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
        self.events = []
        self.agents = []
        
    def add_event(self, event):
        self.events.append(event)
        
    def add_agent(self, agent):
        self.agents.append(agent)
        #add the callback for actuators!
        for actuator in agent.actuators:
            actuator._callback = self.add_event
            
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
        assert len(agent.actuators) == 1, 'An OpenAI-gym simulator only permits agents with a single actuator.'
        assert len(self.agents) <= 1, 'An OpenAi-gym simulator only permits single agent environments.'
        super(GymSimulator, self).add_agent(agent)
        
    def __iter__(self):
        self.running = True
        state = self.__reset__env__()
        yield self.time
        while(self.running):
            self.agents[0].attempt(state)
            action = self.events.pop()
            action = action % self.env.action_space.n #TODO continuous actions
            nstate, reward, self.time.done, _ = self.env.step(action)
            if self.render:
                self.env.render()
            self.time.step += 1
            self.time.global_step += 1
            for sensor in self.agents[0].sensors:
                sensor((state, action, reward, nstate, self.time))
            if self.debug:
                self.debug(self.agents[0], (state, action, reward, nstate, self.time))
            state = nstate
            yield self.time
            if(self.time.done):
               state = self.__reset__env__()
               yield self.time
               
    def __reset__env__(self):
        self.time.episode += 1
        self.time.step = 0
        self.time.done = False
        state = self.env.reset()
        if self.render:
            self.env.render()
        return state