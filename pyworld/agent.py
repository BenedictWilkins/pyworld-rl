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

from . import common as c

import random

from collections import namedtuple

arsi = namedtuple('observation', ['action', 'reward', 'state', 'info'])
sarsi = namedtuple('observation', ['pstate', 'action', 'reward', 'state', 'info'])
rst = namedtuple('observation', ['state', 'info'])

''' *********************************************** AGENTS *********************************************** '''

def ag_sense_callback(sense, attempt):
    while True:
        obs = yield
        sense(obs)
        attempt()
        
class Agent(ABC):

    def __init__(self):
        self._sense_callback = ag_sense_callback(self.sense, self.attempt)
        self._sense_callback.send(None)
        self._reset_callback = ag_sense_callback(self.reset, self.attempt)
        self._reset_callback.send(None)

    def add_component(self, label, component):
        setattr(self, label, component)
        
    def has_component(self, label):
        return hasattr(self, label)
    
    @abstractmethod
    def attempt(self, *args):
        pass
    
    @abstractmethod
    def sense(self, *obs):
        pass
    
    @abstractmethod
    def reset(self, *obs):
        pass
            
class SimpleAgent(Agent):
    
    def __init__(self, sensor, actuator):
        super(SimpleAgent, self).__init__()
        self.sensor = sensor
        self.sensor.register(self)
        self.actuator = actuator
        
    def attempt(self):
        self.actuator()
    
    def reset(self, obs):
        print(obs.info.name, obs.info.step)
    
    def sense(self, obs):
        print(obs.info.name, obs.info.step)
    
    
class GymAgent(Agent):
    
    def __init__(self, sensors, actuators):
        super(GymAgent, self).__init__()
        for sensor in sensors:
            sensor.register(self)
        
        

''' *********************************************** SENSORS *********************************************** '''

def sense_callback(sense, callback):
    while True:
        obs = yield
        for ob in sense(obs):
            callback.send(ob)

class Sensor(ABC):
    
    def __init__(self, sensor=None):
        self.sensor = sensor
        
    @abstractmethod
    def sense(self, obs):
        pass
    
    @abstractmethod
    def reset(self, obs):
        pass
    
    def register(self, agent):
        if self.sensor is not None:
            self.sensor.register(agent)
        else:
            self.sensor = agent
        self.__register(self.sensor)
            
    def __register(self, sensor):
        print("REG:", sensor)
        self._sense_callback = sense_callback(self.sense, sensor._sense_callback)
        self._sense_callback.send(None)
        self._reset_callback = sense_callback(self.reset, sensor._reset_callback)
        self._reset_callback.send(None)
        if isinstance(sensor, Sensor):
            self.observation_space = obs_union(sensor.observation_space, self.observation_space)

def obs_union(l1, l2):
    try:
        index = l1.index(-1)
    except:
        return l1
    l1.pop(index)
    for e in reversed(l2):
        l1.insert(index, e)
    return l1
        
class SimpleSensor(Sensor):
    
    def __init(self):
        super(SimpleSensor, self).__init__()
        
    def sense(self, obs):
        yield obs
        
    def reset(self, obs):
        yield obs
   
'''  
              
class EpisodicSensor(BatchSensor):

        The EpisodicSensor collects actions, rewards and states in a 
        batch whose size is the number of steps of a given episode. 
        This type of sensor may be used for a naive REINFORCE agent.
        Note: The last (terminal) state of an episode is ignored by this sensor.

    
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
'''            
            
class MaxPoolSensor(Sensor):
    
    def __init__(self, sensor=None, skip=3):
        self.observation_space = sensor.observation_space
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
        action, reward, state, info = obs
        self.count += 1
        self.max_pool.append(state)
        self.total_reward += reward
        if self.count % self.skip == 0 or info.done:
            max_frame = np.max(np.stack(self.max_pool), axis=0)

            yield arsi(action, self.total_reward, max_frame, info)
            self.total_reward = 0

class AtariImageSensor(Sensor):
    
    def __init__(self, sensor=None):
         self.observation_space = [84, 84]
         super(AtariImageSensor, self).__init__(sensor)

    def sense(self, obs):
        action, reward, state, time = obs
        tstate = self.transform(state)
        yield arsi(action, reward, tstate, time)
        
    def reset(self, obs):
        state, time = obs
        tstate = self.transform(state)
        yield rst(tstate, time)
        
    def transform(self, state):
        # if state.size == 210 * 160 * 3:
        #   img = np.reshape(state, [state.shape[1], state.shape[2], state.shape[0]]).astype(np.float)
        # elif state.size == 250 * 160 * 3:
        #   img = np.reshape(state, [250, 160, 3]).astype(np.float)
        # else:
        #      assert False, "Unknown resolution."
        img = (state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114).astype(np.float)
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        return np.reshape(img[18:102, :], [1, 84, 84]).astype(np.float) / 255.0

def sensor(name, *methods):
    '''
        Sensor factory
    '''
    assert(len(methods) > 1) #sensor must contain atleast the sense method
    if isinstance(methods[0], tuple):
        return type(name, (Sensor,), {m.__name__:m for m in methods})

class BufferedSensor1(Sensor):
    
    '''
        A sensor that stores states in a circular buffer of a given length. Each cycle 
        an agent will be provided with the buffer that includes the current environment
        state and a number of past states. This kind of sensor can be used to make
        the observation space approximately Markovian. 
        
        Observations take the form:
            <action, reward, buffer, time>
        where action is the action performed by the agent that gave the transition 
        to the current environment state, reward is the reward received at this 
        transtion and buffer is the buffer of states which includes the current 
        and a number of past states, buffer = [s_t-n, ..., s_t-1, s_t].
        
        See also BufferedSensor2
    '''
    
    def __init__(self, sensor=None, stack=4):
        self.observation_space = [stack, -1]
        super(BufferedSensor1, self).__init__(sensor)
        self.buffer = collections.deque(maxlen=stack)
        self.stack = stack
        
    
    def reset(self, obs):
        state, time = obs
        state = state.squeeze()
        for i in range(self.stack - 1):
            self.buffer.append(np.zeros_like(state))
        self.buffer.append(state) 
        yield rst(np.stack(self.buffer), time)
    
    def sense(self, obs):
        action, reward, state, time = obs
        self.buffer.append(state.squeeze())
        yield arsi(action, reward, np.stack(self.buffer), time)

class BufferedSensor2(Sensor):
    '''
        A sensor that stores states in a circular buffer of a given length. Each cycle 
        an agent will be provided with the buffer that includes the current environment
        state and a number of past states. This kind of sensor can be used to make
        the observation space approximately Markovian. 
        
        Observations take the form:
            <pbuffer, action, reward, buffer, time>
        where action is the action performed by the agent that gave the transition 
        to the current environment state, reward is the reward received at this 
        transtion, buffer is the buffer of states which includes the current 
        and a number of past states and pbuffer is a buffer of states which does not include the current state,
        i.e. the buffer of past states.
        buffer = [s_t-n, ..., s_t-1, s_t]
        pbuffer = [s_t-(n+1), ... s_t-2, s_t-1]
        
        This type of observation is useful for the DeepQ learning algorithm.
        
        See also BufferedSensor2
    '''
    def __init__(self, sensor=None, stack=4):
        self.observation_space = [stack, -1]
        super(BufferedSensor2, self).__init__(sensor)
        self.buffer = collections.deque(maxlen=stack)
        self.state = None
        self.stack = stack
    
    def reset(self, obs):
        state, time = obs
        state = state.squeeze()
        for i in range(self.stack - 1):
            self.buffer.append(np.zeros_like(state))
        self.buffer.append(state)
        self.state = np.stack(self.buffer)
        yield rst(self.state, time)
    
    def sense(self, obs):
        action, reward, state, time = obs
        self.buffer.append(state.squeeze())
        nstate = np.stack(self.buffer)
        yield sarsi(self.state, action, reward, nstate, time)
        self.state = nstate
  
class VisualiseSensor(Sensor):
    
    def __init__(self, sensor=None, wait=15, name='state', path=None, step=50, scale=1, split_channels=False):
        self.observation_space = [-1]
        super(VisualiseSensor, self).__init__(sensor)
        self.vis = True
        self.wait = wait
        self.name = name
        self.path = path
        self.extension = '.png'
        self.step = step
        self.scale = scale
        self.split_channels = split_channels
        
    def reset(self, obs):
        print(obs.state.shape)
        
        if self.vis:
            state = self.process(obs.state, self.scale)
            cv2.imshow(self.name, state)
            if obs.info.global_step % self.step == 0:
                if self.path is not None:
                    cv2.imwrite(self.path + self.name + str(obs.info.global_step) + self.extension, state)
            if cv2.waitKey(self.wait) == ord('q'):
                self.vis = False
        yield obs
        
        
    def sense(self, obs):
         if self.vis: 
            state = self.process(obs.state, self.scale)
            cv2.imshow(self.name, state)
            if obs.info.global_step % self.step == 0:
                if self.path is not None:
                    cv2.imwrite(self.path + self.name + str(obs.info.global_step) + self.extension, state)
            if cv2.waitKey(self.wait) == ord('q'):
                self.vis = False
         yield obs
        
    def process(self, state, scale):
        state = state.squeeze()
        shape = state.shape
        if len(shape) == 2:
            if state.dtype == np.float:
                state = (state * 255.).astype(np.uint8)
            state = cv2.cvtColor(state[:,:,np.newaxis],cv2.COLOR_GRAY2RGB)
            state = cv2.resize(state, (0,0), interpolation=cv2.INTER_AREA, fx=scale, fy=scale)
        elif len(shape) == 3:
            if shape[2] > shape[0]: #put channel in the correct place - this is not always true... but what can you do!
                state = np.swapaxes(np.swapaxes(state, 2, 0), 0,1)
            if self.split_channels:
                #split the image into a number of grayscale channels
                rt = tuple([state[:,:,i] for i in range(state.shape[2])])
                state = np.concatenate(rt, axis=1)
                print(state.shape)
                state = self.process(state, scale) #process the new grayscale image
        return state

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

class ProbabilisticActuator(Actuator):
    
    def __init__(self, softmax=True):
        super(ProbabilisticActuator, self).__init__()
        if softmax:
            self.attempt = self.__sm_attempt
        else:
            self.attempt = self.__attempt
    
    def __attempt(self, action_probs):
        action_probs = action_probs.squeeze()
        action = np.random.choice(len(action_probs), p=action_probs)
        return action
    
    def __sm_attempt(self, action_probs):
        action_probs = action_probs.squeeze()
        eap = np.exp(action_probs)
        action_probs = eap / sum(eap)
        action_probs = action_probs
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
        scores = scores.squeeze()
        epsilon = next(self.epsilon_tracker)
        if random.random() > epsilon:
            action = np.argmax(scores)
        else:
            action = np.random.choice(len(scores))
        return action

class EpsilonSchedule():
    def __init__(self, **params):
        super(EpsilonSchedule, self).__init__(enabled=True)
        self.epsilon_start = params.get('epsilon_start', 1.0)
        self.epsilon_final = params.get('epsilon_final', 0.01)
        self.epsilon_frames = params.get('epsilon_frames', 10**5)
        self.epsilon = self.epsilon_start
        
    def get(self):
        return self.epsilon
        
    def episode(self, _):
        pass
    
    def step(self, frame):
        self.epsilon = max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames) 
        
    def __iter__(self):
        frame = 0
        while not self.epsilon == self.epsilon_final:
            frame += 1
            self.step(frame)
            yield self.epsilon
        while(True):
            yield self.epsilon
            

''' *********************************************** EXPERIENCE *********************************************** '''
class Experience:
    pass
               
class BatchExperience(Experience):
    
    def __init__(self, batch_labels = ['action', 'reward', 'state'], batch_size = 16):
        self.batch_labels = batch_labels
        self._batch = c.batch(batch_labels)
        self.buffer = self._batch(*[[] for _ in batch_labels])
        self.batch_size = batch_size

    def append(self, *obs):
        for i in range(len(obs)):
            self.buffer[i].append(obs[i])
            
    def extend(self, obs):
        for i in range(len(self.buffer)):
            self.buffer[i].extend(obs[i])
    
    def full(self):
        return len(self.buffer[0]) > self.batch_size
    
    def pop(self):
        popped = [x[:self.batch_size] for x in self.buffer]
        for i in range(len(self.buffer)):
            self.buffer[i] = self.buffer[i][self.batch_size:]
        return self._batch(*popped)
    
    def popall(self):
        pop = self._batch(*self.buffer)
        self.buffer = self._batch(*[[] for _ in self.batch_labels])
        return pop
        
    def clear(self):
        for s in self.buffer:
            s.clear()
                    
    def batch_to_numpy(self, batch, types, copy=False):
        return [np.array(batch[i], copy=copy, dtype=types[i]) for i in range(len(batch))]

    def batch_to_tensor(self, batch, types, device='cpu'):
        return [types[i](batch[i]).to(device) for i in range(len(batch))]
    

class UnrollExperience(BatchExperience):
    BATCH_LABELS = ['state', 'action', 'reward']
    
    def __init__(self, batch_update, batch_labels = BATCH_LABELS, gamma=0.99, batch_size=16, steps=-1):
        super(UnrollExperience, self).__init__(batch_labels=batch_labels, batch_size=batch_size)
        empty_batch = lambda: self._batch(*[[] for _ in batch_labels])
        if steps > 0:
            self.acc = UnrollExperience.UnrollAccStep(gamma, empty_batch, batch_update, steps)
        else:
            self.acc = UnrollExperience.UnrollAccEpisode(gamma, empty_batch, batch_update)
                
    def append(self, *obs):

        self.acc.update(*obs, self.__exp)
        
    def __exp(self, obs):
        super(UnrollExperience, self).extend(obs)
            
    class UnrollAccEpisode:
        
        def __init__(self, gamma, empty_batch, batch_update):
            self.observations = empty_batch()
            self.gamma = gamma
            self.empty_batch = empty_batch
            self.batch_update = batch_update

        
        def update(self, obs, exp):
            #assert(self.observations._fields == obs._fields) #disable optimisation!
            self.batch_update(self.observations, obs)
            if obs.info.done:
                #compute discounted reward
                rewards = self.observations.reward
                dr = 0
                for i in range(len(rewards)-1, -1, -1):
                    rewards[i] = rewards[i] + self.gamma * dr
                    dr = rewards[i]
                exp(self.observations)
                #reset observations
                self.observations = self.empty_batch()
                
    class UnrollAccStep:
        
        def __init__(self, gamma, empty_batch, batch_update, steps=2):
            empty = empty_batch()
            self.steps = steps + 1
            self.observations = empty_batch(*[collections.deque(maxlen=self.steps) for i in range(len(empty))])
            self.gamma = gamma
            self.empty_batch = empty_batch
            self.update = self.__init_update
            self.batch_update = batch_update
            
        def __init_update(self, obs, exp):
            #assert(self.observations._fields == obs._fields) #disable optimisation!
            self.batch_update(self.observations, obs)
            
            #be careful not to have a large step, this could cause problems with repreated multiplication..
            gg = self.gamma
            for i in range(len(self.observations.reward)-2, -1, -1):
                self.observations.reward[i] += (gg * obs.reward)
                gg *= self.gamma
                
            if(len(self.observations.reward) == self.steps):
                self.update = self.__update
                exp(self.empty_batch(*[[o.popleft()] for o in self.observations])) #never done here
                
            
            
        def __update(self, obs, exp):
            #assert(self.observations._fields == obs._fields) #disable optimisation!
            self.batch_update(self.observations, obs)

            gg = 1 * self.gamma
            for i in range(len(self.observations.reward)-2, -1, -1):
                self.observations.reward[i] += (gg * obs.reward)
                gg *= self.gamma
            print("u", self.observations)
            exp(self.empty_batch(*[[o.popleft()] for o in self.observations])) #never done here
            print("u", self.observations)
            
            if obs.info.done:
                for i in range(len(self.unroll_states)):
                    exp(self.empty_batch(*[[o.popleft()] for o in self.observations]))
                self.update = self.__init_update

   
          
class ExperienceReplay(Experience):
    
    def __init__(self, size = 100000, initial_size = 10000):
        self.buffer = collections.deque(maxlen=size)
        self.size = size
        self.initial_size = initial_size
        
    def append(self, *exp):
        self.buffer.append(exp)
        
    def sample(self, batch_size = 64):
        sample_size = min(batch_size, len(self.buffer))
        return list(map(list, zip(*random.sample(self.buffer, k=sample_size))))
   
    def initial_full(self):
        return len(self.buffer) >= self.initial_size
    
    def full(self):
        return len(self.buffer) == self.buffer.maxlen
    
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



if __name__ == "__main__":
    

    from simulate import Meta as meta
    
    def batch_update(batch, obs):
        batch.reward.append(obs.reward)
        batch.step.append(obs.info.step)
    
    def testEpisodeUnrollExperience():
        ob = collections.namedtuple('observation', 'reward info')
        exp = UnrollExperience(batch_update, batch_labels=['reward', 'info'], )
        info = meta(0)
        exp.append(ob(1,info))
        info.done = True
        exp.append(ob(1,info))
        reward,info = exp.pop()
        print(info)
        print(reward)
        
    def testStepUnrollExperience():
        ob = collections.namedtuple('observation', 'reward info')
        exp = UnrollExperience(batch_update, batch_labels=['reward', 'step'], steps=2)
        info = meta(0)
        exp.append(ob(1,info))
        info.step += 1
        exp.append(ob(1,info))
        info.step += 1
        exp.append(ob(1,info))
        info.step += 1
        exp.append(ob(1,info))
        

        print(exp.pop())
        print(exp.pop())
        
    testStepUnrollExperience()
    #testEpisodeUnrollExperience()
    
    
    
    
    
    
    
    
    
    
    
    
    
    