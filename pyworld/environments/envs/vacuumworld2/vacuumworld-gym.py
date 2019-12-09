#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:55:54 2019

author: Benedict Wilkins
"""

import vacuumworld 
import vacuumworld.vwenvironment as vwenv
import vacuumworld.vwc as vwc
import vacuumworld.vwsensor as vwsensor
import vacuumworld.vw as vw
import vacuumworld.vwutils as vwutils
import vacuumworld.saveload as saveload

import numpy as np
import typing

import gym

class observation(typing.NamedTuple): 
    
    state : np.ndarray
    
class ObservationProcess(vwenv.ObservationProcess):
    
    colour_table = {
                vwc.colour.green:1,
                vwc.colour.orange:2,
                vwc.colour.white:3,
                vwc.colour.user:4}
    orientation_table = {
            vwc.orientation.north:1,
            vwc.orientation.east:2,
            vwc.orientation.south:3,
            vwc.orientation.west:4}
    
    conversion_table = {
            vwc.agent:lambda agent: ObservationProcess.conversion_table[vwc.colour](agent.colour) + 4 * ObservationProcess.conversion_table[vwc.orientation](agent.orientation), #1,2,3,4 + 4 * 1,2,3,4
            vwc.dirt:lambda dirt: ObservationProcess.conversion_table[vwc.colour](dirt.colour), #1,2
            vwc.colour:lambda colour: ObservationProcess.colour_table[colour], #1,2,3,4
            vwc.orientation:lambda orien: ObservationProcess.orientation_table[orien]} #1,2,3,4
    
    def __init__(self, dtype=np.int):
        self.dtype = dtype
        if dtype==np.int:
            self.convert = self.to_int
        else:
            self.convert = self.to_float
    
    def to_float(self, arg):
        return ObservationProcess.conversion_table[type(arg)](arg) / 20.
    
    def to_int(self, arg):
        return ObservationProcess.conversion_table[type(arg)](arg)
            
    def get_perception(self, grid, agent):
        #convert grid to numpy array
        #CHW - torch format
        obs = np.zeros((3, grid.dim, grid.dim), dtype=self.dtype)

        c = agent.coordinate
        obs[0, c[0], c[1]] = 1 #mask for the current agent
        
        for c,a in grid._get_agents().items(): #all agents
            obs[1, c[0],c[1]] = self.convert(a)
            
        for c,d in grid._get_dirts().items(): #all dirts
            obs[2, c[0], c[1]] = self.convert(d)

        return observation(obs)
    
    
class ObservationProcessSimple(vwenv.ObservationProcess):
    
    '''
        Observation process for a single grid configuration with a single agent.
    '''
    
    def get_perception(self, grid, agent):
        return agent.coordinate #this is enough to capture the full state (everything else is implicit!)

#change the observation process to provide the full grid as an observation
vwenv.ObservationProcess = ObservationProcess 
#change the subscription to the new type of observation
vwsensor.VisionSensor.subscribe = [observation]
            
        
        
 

#move this somewhere more general...


actions = [vwc.action.move(), vwc.action.clean(), vwc.action.turn(vwc.direction.left), vwc.action.turn(vwc.direction.right), vwc.action.idle()]
action_space = gym.spaces.Discrete(len(actions))

from pyworld.algorithms.rl.TabularQLearning import TabularQLearning
    
class QLearningMind:
    
    def __init__(self, action_space, qlearn = None, t=None, train=True):
        self.state = 0
        self.action = 0
        self.reward = 0
        self.train = train
        if qlearn is not None:
            self.qlearn = qlearn 
            if t is not None:
                self.qlearn.boltzman_dist.t = t
        else:
            if t is not None:
                self.qlearn = TabularQLearning(action_space, temp=t)
            else:
                self.qlearn = TabularQLearning(action_space)
        self.qlearn_update = self.__qlui
        
        self.cycle = 0
    
    def reset(self):
        self.qlearn_update = self.__qlui
        
    def __qlui(self, *_):
        self.qlearn_update = self.__qlu
        
    def __qlu(self, state, action, next_state):        
        reward = self.intrinsic_reward(state, action, next_state)
        b_next_state = bytes(next_state.data)
        self.qlearn.update(bytes(state.data), action, reward, b_next_state)
        
        
        if not self.train:
            action_probs, values = self.qlearn.action_probs(b_next_state)
            print(" ".join(["%.4f" % x for x in action_probs]))
            print(values)

    def intrinsic_reward(self, state, action, next_state):
        if action == 1.: #clean
            position = np.where(state[0] == 1)
            dirt = state[2, position[0], position[1]]
            if dirt > 0: #there was some dirt
                colour = state[1,position[0],position[1]] % 4
                if colour == dirt or colour == 3: #it was the correct colour!
                    return 10.
                
        return 0.
    
    def decide(self):
        self.action = self.qlearn.policy(bytes(self.state))
        #print(self.action)
        return actions[self.action]

    def revise(self, observation, _):
        self.qlearn_update(self.state, self.action, observation.state)
        self.state = observation.state
        
        #print(self.qlearn.action_probs(bytes(observation.state.data)))


import pyworld.toolkit.tools.fileutils as fu
import copy

def train(white_mind, green_mind=None, orange_mind=None, cont=False, steps=1000):
    minds = {}
    minds[vwc.colour.white], minds[vwc.colour.green], minds[vwc.colour.orange] = vwutils.process_minds(white_mind, green_mind, orange_mind)
    
    if not cont:
        qlearn = TabularQLearning(action_space)
    else:
        qlearn = fu.load("test.pickle")
    
    #dim = 4
    #grid = vw.random_grid(dim,1,0,0,0,np.random.randint(0,dim*dim),0)
    grid = saveload.load('test.vw')
    
    for i in range(0, steps):
        
        env = vwenv.GridEnvironment(vwenv.GridAmbient(copy.deepcopy(grid), minds))
        
        list(env.ambient.agents.values())[0].mind.surrogate.qlearn = qlearn #sigh... avoid a large copy
        
        
        
        #print(env.ambient.agents)
        
        
        #vacuumworld.run(mind, skip=True, play=True, load='test.vw', speed=0)
        print("episode:", i, "dirts: ", len(env.ambient.grid._get_dirts()))
        #for i in range(2):
        while len(env.ambient.grid._get_dirts()) != 0:
            env.evolveEnvironment()
            
        if not i % 100:
            fu.save("test.pickle", qlearn)

            
mind = QLearningMind(action_space, t=0.5) 
#train(mind)


mind = QLearningMind(action_space, qlearn=fu.load("test.pickle"), t = 0.5, train=False)  
vacuumworld.run(mind, skip=True, load='test.vw', play=True)





