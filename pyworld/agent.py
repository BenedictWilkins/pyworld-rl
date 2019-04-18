#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:21:27 2019

@author: ben
"""
import numpy as np
from abc import ABC, abstractmethod
import collections
import cv2
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
    def __reset__(self, *args):
        pass

    @abstractmethod
    def __call__(self, *args):
        pass
    
class SimpleSensor(Sensor):
    
    def __init(self):
        super(SimpleSensor, self).__init__()
    
    def __call__(self, obs):
        self._callback(obs)
        
    def __reset__(self, obs):
        state, time = obs
        self._callback((0.0, None, state, time))

    
class BatchSensor(Sensor):
    
    def __init__(self, batch_labels = ['action', 'reward', 'state'], batch_size = 16, callback=None):
        super(BatchSensor, self).__init__(callback)
        Batch = c.batch(batch_labels)
        self.batch = Batch()
        self.batch_size = batch_size
    
    def batch_append(self, *obs):
        for i in range(len(obs)):
            self.batch[i].append(obs[i])
    
    def batch_full(self):
        return len(self.batch[0]) == self.batch_size
    
    def batch_clear(self):
        for s in self.batch:
            s.clear()
        
              
class EpisodicSensor(BatchSensor):
    '''
        The EpisodicSensor collects actions, rewards and states in a 
        batch whose size is the number of steps of a given episode. 
        This type of sensor may be used for a naive REINFORCE agent.
        Note: The last (terminal) state of an episode is ignored by this sensor.
    '''
    
    def __init__(self, callback=None):
       super(EpisodicSensor, self).__init__(batch_labels=['state', 'action', 'reward', 'time'], callback=callback)
       self.total_reward = 0
       self.state = None
    
    def __reset__(self, obs):
        state, time = obs
        self.state = state
        
    def __call__(self, obs):
        action, reward, state, time = obs
        
        self.batch_append(self.state, action, reward, time)
        self.state = state
        
        if time.done:
            self._callback(self.batch)
            self.batch_clear()
        
                 
class UnrollSensor(BatchSensor):
    
    def __init__(self, batch_size = 16, callback=None, gamma=0.99, steps=3):
        super(UnrollSensor, self).__init__(batch_labels=['state', 'action', 'reward', 'nstate', 'time'], callback=callback)
        self.gamma =  gamma
        self.steps = steps
        self.total_reward = 0
        self.unroll_states = [None] * steps
        self.unroll_actions = [None] * steps
        self.unroll_times = [None] * steps
        self.unroll_rewards = np.zeros(steps)
        self.state = None

    def __reset__(self, obs ):
        state, time = obs
        self.state = state

    def __call__(self, obs):
        action, reward, state, time = obs
        #print(pstate, time)
        
        i = time.step % self.steps
        
        #print(pstate, time)
        if self.unroll_states[i] is not None:
            self.batch_append(self.unroll_states[i], self.unroll_actions[i], self.unroll_rewards[i], self.state, self.unroll_times[i])
       
        self.unroll_states[i] = self.state
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
                    self.batch_append(self.unroll_states[k], self.unroll_actions[k], self.unroll_rewards[k], self.state, self.unroll_times[k])

            self.unroll_states = [None] * self.steps
            self.unroll_actions = [None] * self.steps
            self.unroll_times = [None] * self.steps
            self.unroll_rewards = np.zeros(self.steps)
        
        self.state = state
        
    def batch_append(self, *args):
        super(UnrollSensor, self).batch_append(*args)
        if self.batch_full():
            self._callback(self.batch)
            self.batch_clear()
        
        
        
    
class AtariSensor(Sensor):
    
    def __init__(self, callback, frames=4):
        super(AtariSensor, self).__init__(callback)
        self.transform = AtariTransform()
        self.buffer = collections.deque([0]*frames, maxlen = frames)
        self.state = None
        
    def __reset__(self, obs):
        state, _ = obs
        self.buffer.append(state)
        self.state = np.array(self.buffer)
        
    def __call__(self, obs):
        action, reward, state, time = obs
        self.buffer.append(state)
        nstate = np.array(self.buffer)
        self._callback((state, action, reward, nstate, time))
        self.state = nstate

class StateTransform(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, state):
        pass
    
class AtariTransform(StateTransform):
    
    def __init__(self, out_shape=[1,84,84]):
        self.in_shape = [-1, -1, 3]
        self.out_shape = out_shape
    
    def __call__(self, state):
        if state.size == 210 * 160 * 3:
             img = np.reshape(state, [210, 160, 3]).astype(np.float32)
        elif state.size == 250 * 160 * 3:
            img = np.reshape(state, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114) 
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        return np.reshape(img[18:102, :], [1, 84, 84]) / 255.0
        

    
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

    
