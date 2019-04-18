#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:21:27 2019

@author: ben
"""
import numpy as np
from abc import ABC, abstractmethod
import copy
from . import common as c

''' *********************************************** AGENTS *********************************************** '''
class Agent(ABC):
    
    def __init__(self, sensors=None, actuators=None):
        self.sensors = sensors
        self.actuators = actuators
        print(self.sensors)
        print(self.actuators)
        
        #debugging info
        self.info = {}
        self.summary_info = {}
        self.update_summary = False
    
    @abstractmethod
    def attempt(self):
        pass

    @abstractmethod
    def sense(self, obs):
        pass
            
            
class LearningAgent(Agent):
    
    def __init__(self, model, optimizer, sensors=None, actuators=None):
        super(LearningAgent, self).__init__(sensors, actuators)
        self.optimizer = optimizer
        self.model = model
         
    @abstractmethod
    def loss(self):
        pass
    
    @abstractmethod
    def train(self, batch):
        pass
    
''' *********************************************** SENSORS *********************************************** '''

class Sensor(ABC):
    
    def __init__(self, callback=None):
        self._callback = callback

    @abstractmethod
    def __call__(self, *args):
        pass
    
class SimpleSensor(Sensor):
    
    def __init(self):
        super(SimpleSensor, self).__init__()
    
    def __call__(self, obs):
        self._callback(obs)

    
class BatchSensor(Sensor):
    
    batch_labels = ['pstate', 'action', 'reward', 'state', 'done']
    
    def __init__(self, batch_size = 16, callback=None):
        super(BatchSensor, self).__init__(callback)
        Batch = c.batch(BatchSensor.batch_labels)
        self.batch = Batch()
        self.batch_size = batch_size

        
    def __call__(self, obs):
        (pstate, action, reward, state, time) = obs
        #print(pstate, reward, action, time)
        self.batch.pstate.append(pstate)
        self.batch.action.append(action)
        self.batch.state.append(state)
        self.batch.reward.append(reward)
        self.batch.done.append(time.done)
       
        if time.global_step % self.batch_size == 0:
            self._callback(self.batch)
            self.batch.pstate.clear()
            self.batch.action.clear()
            self.batch.reward.clear()
            self.batch.state.clear()
            self.batch.done.clear()
            
              
class EpisodicSensor(Sensor):
    '''
        The EpisodicSensor collects actions, rewards and states in a 
        batch whose size is the number of steps of a given episode. 
        This type of sensor may be used for a naive REINFORCE agent.
        Note: The last (terminal) state of an episode is ignored by this sensor.
    '''
    
    batch_labels = ['action', 'reward', 'state']
    
    def __init__(self, callback=None):
       super(EpisodicSensor, self).__init__(callback)
       Batch = c.batch(EpisodicSensor.batch_labels)
       self.batch = Batch()
       self.total_reward = 0
    
    def __call__(self, obs):
        (pstate, action, reward, _, time) = obs
      
        self.batch.action.append(action)
        self.batch.reward.append(reward)
        self.batch.state.append(pstate)
        
        if time.done:
            self._callback(self.batch)
            self.batch.action.clear()
            self.batch.reward.clear()
            self.batch.state.clear()
        
                 
class UnrollSensor(BatchSensor):
    
    def __init__(self, batch_size = 16, callback=None, gamma=0.99, steps=3):
        super(UnrollSensor, self).__init__(batch_size, callback)
        self.gamma =  gamma
        self.steps = steps
        self.total_reward = 0
        self.unroll_states = [None] * steps
        self.unroll_actions = [None] * steps
        self.unroll_times = [None] * steps
        self.unroll_rewards = np.zeros(steps)


    def __call__(self, obs):
        pstate, action, reward, state, time = obs
        #print(pstate, time)
        
        i = time.step % self.steps
        
        r_state = self.unroll_states[i]
        r_action = self.unroll_actions[i]
        r_time = self.unroll_times[i]
        r_reward = self.unroll_rewards[i]
        #print(pstate, time)
        if r_state is not None:
            #print('i:', i, r_reward)
            super(UnrollSensor, self).__call__((r_state, r_action, r_reward, pstate, r_time))
            #self.callback((r_state, r_action, r_reward, pstate, r_time))
       
        self.unroll_states[i] = pstate
        self.unroll_actions[i] = action
        self.unroll_times[i] = copy.copy(time)
        self.unroll_rewards[i] = 0
        #move all this above end?
        #compute the unrolled discounted reward
        gg = 1
        for j in range(self.steps):
            self.unroll_rewards[(i + self.steps - j) % self.steps] += (gg * reward)
            gg *= self.gamma
        
        #provide all of the unrolled rewards capped at the end time to the agent and reset
        if time.done:
            #print(self.unroll_states)
            #print(self.unroll_rewards)
            for j in range(i + 1, i + self.steps + 1):
                k = j % self.steps
                if self.unroll_states[k] is not None:
                   # print(k, self.unroll_rewards[k])
                    #self._callback((self.unroll_states[k], self.unroll_actions[k], self.unroll_rewards[k], pstate, self.unroll_times[k]))
                    super(UnrollSensor, self).__call__((self.unroll_states[k], self.unroll_actions[k], self.unroll_rewards[k], pstate, self.unroll_times[k]))
            self.unroll_states = [None] * self.steps
            self.unroll_actions = [None] * self.steps
            self.unroll_times = [None] * self.steps
            self.unroll_rewards = np.zeros(self.steps)
        
       
        
        
            
            
        

    
''' *********************************************** ACTUATORS *********************************************** '''

class Actuator(ABC):
    
    def __init__(self, callback=None):
        self._callback = callback

    @abstractmethod
    def __call__(self, *args):
        pass
    
class SimpleActuator(Actuator):
    def __init(self):
        super(SimpleActuator, self).__init__()
        
    def __call__(self, action):
        self._callback(action)
 

class RandomActuator(Actuator):
    
    def __init__(self, action_space, callback=None):
        super(RandomActuator, self).__init__(callback)
        self.action_space = action_space
        
    def __call__(self):
        self._callback(self.action_space.sample())

class ProbabilisticActuator:
    
    def __init__(self, callback=None):
        super(ProbabilisticActuator, self).__init__()
    
    def __call__(self, action_probs):
        action_probs = action_probs.detach().numpy()
        action = np.random.choice(len(action_probs), p=action_probs)
        self._callback(action)

    
