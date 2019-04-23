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

import random

from collections import namedtuple

arst = namedtuple('arst', ['action', 'reward', 'state', 'time'])
sarst = namedtuple('sarst', ['pstate', 'action', 'reward', 'state', 'time'])

''' *********************************************** AGENTS *********************************************** '''
class Agent(ABC):

    def __init__(self):
        pass
    
    def add_component(self, label, component):
        setattr(self, label, component)
    
    @abstractmethod
    def attempt(self):
        pass

    @abstractmethod
    def sense(self, obs):
        pass
    
    def reset(self, obs):
        pass
    
    def __sense__(self):
        while(True):
            obs = yield
            self.sense(obs)
            self.attempt()
            
    def __reset__(self):
         while(True):
            obs = yield
            self.reset(obs)
            self.attempt()

    
''' *********************************************** SENSORS *********************************************** '''

class Sensor(ABC):
    
    def __init__(self, sensor):
        assert(isinstance(sensor, Sensor) or isinstance(sensor, Agent))
        self.sensor = sensor
        self.sense_callback = sensor.__sense__()
        self.sense_callback.send(None)
        self.reset_callback = sensor.__reset__()
        self.reset_callback.send(None)

    @abstractmethod
    def sense(self, obs):
        pass
    
    @abstractmethod
    def reset(self, obs):
        pass
    
    def __sense__(self):
        while(True):
            obs = yield
            for obs in self.sense(obs):
                self.sense_callback.send(obs)
                
    def __reset__(self):
         while(True):
            obs = yield
            for obs in self.reset(obs):
                self.reset_callback.send(obs)
        
class SimpleSensor(Sensor):
    
    def __init(self, info):
        super(SimpleSensor, self).__init__()
        self.info = info
        
    def sense(self, obs):
        yield obs
        
    def reset(self, obs):
        yield obs
    
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
            self.__callback(self.batch)
            self.batch_clear()

class UnrollSensor(Sensor):
    
    def __init__(self, batch_size = 16, callback=None, gamma=0.99, steps=3):
        super(BatchUnrollSensor, self).__init__(batch_labels=['state', 'action', 'reward', 'nstate', 'time'], callback=callback)
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
             super(UnrollSensor, self).__callback(obs).__callback((self.unroll_states[i], self.unroll_actions[i], self.unroll_rewards[i], self.state, self.unroll_times[i]))
       
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
                    #self.__callback((self.unroll_states[k], self.unroll_actions[k], self.unroll_rewards[k], pstate, self.unroll_times[k]))
                    super(UnrollSensor, self).__callback((self.unroll_states[k], self.unroll_actions[k], self.unroll_rewards[k], self.state, self.unroll_times[k]))

            self.unroll_states = [None] * self.steps
            self.unroll_actions = [None] * self.steps
            self.unroll_times = [None] * self.steps
            self.unroll_rewards = np.zeros(self.steps)
        
        self.state = state
                 
class BatchUnrollSensor(BatchSensor):
    
    def __init__(self, batch_size = 16, callback=None, gamma=0.99, steps=3):
        super(BatchUnrollSensor, self).__init__(batch_labels=['state', 'action', 'reward', 'nstate', 'time'], callback=callback)
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
                    #self.__callback((self.unroll_states[k], self.unroll_actions[k], self.unroll_rewards[k], pstate, self.unroll_times[k]))
                    self.batch_append(self.unroll_states[k], self.unroll_actions[k], self.unroll_rewards[k], self.state, self.unroll_times[k])

            self.unroll_states = [None] * self.steps
            self.unroll_actions = [None] * self.steps
            self.unroll_times = [None] * self.steps
            self.unroll_rewards = np.zeros(self.steps)
        
        self.state = state
        
    def batch_append(self, *args):
        super(BatchUnrollSensor, self).batch_append(*args)
        if self.batch_full():
            super(BatchUnrollSensor, self).__callback(self.batch)
            self.batch_clear()
        
class MaxPoolSensor(Sensor):
    
    def __init__(self, sensor=None, skip=3):
        super(MaxPoolSensor, self).__init__(sensor)
        self.max_pool = collections.deque(maxlen=skip)
        self.skip = skip
        self.total_reward = 0
        self.count = 0
        
    def reset(self, obs):
        state, _= obs
        self.total_reward = 0
        self.max_pool.clear()
        self.max_pool.append(state)
        yield obs #max pooling here = state
        
    def sense(self, obs):
        action, reward, state, time = obs
        self.count += 1
        self.max_pool.append(state)
        self.total_reward += reward
        if self.count % self.skip == 0 or time.done:
            max_frame = np.max(np.stack(self.max_pool), axis=0)
            #cv2.imshow("max_pool", max_frame)
            yield (action, self.total_reward, max_frame, time)
            self.total_reward = 0

class AtariImageSensor(Sensor):
    

    
    def __init_(self, sensor=None):
         super(AtariImageSensor, self).__init__(sensor)
         self.in_shape = [-1,-1,3]
         self.out_shape = [1, 84, 84]
    
    def sense(self, obs):
        action, reward, state, time = obs
        tstate = self.transform(state)
        yield (action, reward, tstate, time)
        
    def reset(self, obs):
        state, time = obs
        tstate = self.transform(state)
        yield (tstate, time)
        
    def transform(self, state):
        if state.size == 210 * 160 * 3:
             img = np.reshape(state, [210, 160, 3]).astype(np.float)
        elif state.size == 250 * 160 * 3:
            img = np.reshape(state, [250, 160, 3]).astype(np.float)
        else:
            assert False, "Unknown resolution."
        img = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114) 
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        return np.reshape(img[18:102, :], [1, 84, 84]).astype(np.float) / 255.0



class BufferedSensor(Sensor):
    
    def __init__(self, sensor=None, frames=4):
        super(BufferedSensor, self).__init__(sensor)
        self.buffer = collections.deque(maxlen=4)
        self.state = None
        self.frames = frames
    
    def reset(self, obs):
        state, time = obs
        state = state.squeeze()
        for i in range(self.frames - 1):
            self.buffer.append(np.zeros_like(state))
        self.buffer.append(state)
        self.state = np.stack(self.buffer)
        yield (self.state, time)
    
    def sense(self, obs):
        action, reward, state, time = obs
        self.buffer.append(state.squeeze())
        nstate = np.stack(self.buffer)
        yield (self.state, action, reward, nstate, time)
        self.state = nstate
  


    
''' *********************************************** ACTUATORS *********************************************** '''

class Actuator:
    
    def __init__(self):
        self.action = None
    
    def __call__(self, *args):
        self.action = self.attempt(*args)
        
    def __act__(self):
        while(True):
            yield self.action
    
    @abstractmethod
    def attempt(self, *args):
        pass
    
class SimpleActuator(Actuator):
    def __init(self):
        super(SimpleActuator, self).__init__()
        
    def attempt(self, action):
        return action
    
class RandomActuator(Actuator):
    
    def __init__(self, action_space):
        super(RandomActuator, self).__init__()
        self.action_space = action_space
        
    def attempt(self):
        return self.action_space.sample()

class ProbabilisticActuator:
    
    def __init__(self, callback=None):
        super(ProbabilisticActuator, self).__init__()
    
    def attempt(self, action_probs):
        action_probs = action_probs.detach().numpy()
        action = np.random.choice(len(action_probs), p=action_probs)
        return action
        
class GreedyActuator(Actuator):
    
    def __init_(self):
        super(GreedyActuator, self).__init__()
        
    def attempt(self, scores):
        return np.argmax(scores)
        
class EpsilonGreedyActuator(Actuator):

    def __init__(self, epsilon_tracker):
        super(EpsilonGreedyActuator, self).__init__()
        assert(c.isiterable(epsilon_tracker))
        self.epsilon_tracker = iter(epsilon_tracker)
        
    def attempt(self, scores):

        if random.random() > next(self.epsilon_tracker):
            action = np.argmax(scores)
        else:
            action = np.random.choice(len(scores))
        print(action, len(scores))
        return action
        

class EpsilonTracker:
    def __init__(self, **params):
        self.epsilon_start = params.get('epsilon_start', 1.0)
        self.epsilon_final = params.get('epsilon_final', 0.01)
        self.epsilon_frames = params.get('epsilon_frames', 10**5)

    def __iter__(self):
        frame = 0
        while(True):
            frame += 1
            yield max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)

''' *********************************************** EXPERIENCE *********************************************** '''



class ExperienceReplay:
    
    def __init__(self, size = 128):
        self.buffer = collections.deque(maxlen=size)
        
    def append(self, *exp):
        self.buffer.append(exp)
        
    def sample(self, batch_size = 8):
        sample_size = min(batch_size, len(self.buffer))
        return map(list, zip(*random.sample(self.buffer, k=sample_size)))
    
    
class PrioritisedExperienceReplay(ExperienceReplay):
    
    def __init__(self, size = 128):
        super(PrioritisedExperienceReplay, self).__init__(size)
        self.priorities = collections.deque(maxlen=size)
    
    def append(self, *exp):
        self.buffer.append(exp)
        self.priorities.append(max(self.priorities, default=1)) #what is a good initial priority?
        
    def get_probabilities(self, priority_damp):
        damped_priorities = np.array(self.priorities) ** priority_damp
        return  damped_priorities / sum(damped_priorities)
    
    def get_importance(self, probabilities):
        importance = 1. / len(self.buffer) * 1. / probabilities
        importance_norm = importance / max(importance)
        return importance_norm
    
    def sample(self, batch_size = 8, priority_damp=1.0):
        sample_size = min(batch_size, len(self.buffer))
        sample_probs = self.get_probabilities(priority_damp)
        sample_ind = np.random.choice(range(0,len(self.buffer)), size=sample_size, p=sample_probs)
        importance = self.get_importance(sample_probs[sample_ind])
        samples = np.array(self.buffer)[sample_ind]
        return map(np.array, zip(*samples)), importance

        
        







