#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:48:22 2019

author: Benedict Wilkins
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:55:54 2019

author: Benedict Wilkins
"""
        
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
  
from pyworld.algorithms.rl.TabularQLearning import TabularQLearning
#from pyworld.toolkit.tools.visutils import HeatmapAnimation
import pyworld.toolkit.tools.datautils as du
from pyworld.toolkit.tools.debugutils import Time
      
import vacuumworld 
import vacuumworld.vwenvironment as vwenv
import vacuumworld.vwc as vwc
import vacuumworld.vwutils as vwutils
import vacuumworld.vwsensor as vwsensor

import vacuumworld.vwv as vwv
import vacuumworld.saveload as saveload

import numpy as np
import typing

import gym

class observation(typing.NamedTuple): 
    
    coordinate : vwc.coord
    orientation : vwc.orientation
    dirt : tuple

class ObservationProcess(vwenv.ObservationProcess):
    
    '''
        Observation process for a single grid configuration with a single agent.
    '''
    
    def get_perception(self, grid, agent):
        
        
        return observation(agent.coordinate, agent.orientation, tuple(grid._get_dirts().keys())) #this is enough to capture the full state (everything else is implicit!)

#change the observation process to provide the full grid as an observation
vwenv.ObservationProcess = ObservationProcess 
#change the subscription to the new type of observation
vwsensor.VisionSensor.subscribe = [observation]

#move this somewhere more general...

actions = [vwc.action.move(), vwc.action.clean(), vwc.action.turn(vwc.direction.left), vwc.action.turn(vwc.direction.right), vwc.action.idle()]
action_space = gym.spaces.Discrete(len(actions))



class QLearningMind:
    
    def __init__(self, action_space, qlearn = None, t=None, train=True):
        self.state = None #intial observation
        self.action = None
        self.reward = None
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

        self.cycle = 0
        
    def qlearn_update(self, state, action, reward, next_state):        
        self.qlearn.update(state, action, reward, next_state)
        
        if not self.train:
            action_probs, values = self.qlearn.action_probs(next_state)
            print(" ".join(["%.4f" % x for x in action_probs]))
            print(values)
            print(len(self.qlearn.values))
    
    def decide(self):
        self.action = self.qlearn.policy(self.state)
        return actions[self.action]

    def revise(self, state, _):
        if self.state is not None:
            #intrinsic reward
            reward = -1.
            if self.action == 1 and self.state.coordinate in self.state.dirt: #there is only a single dirt/agent colour
                reward = 10.
            
            self.qlearn_update(self.state, self.action, reward, state)
        
        self.state = state
        
        #print(self.qlearn.action_probs(bytes(observation.state.data)))
        if not self.train and self.state is not None:
            global value_grid

            for i in range(value_grid.shape[0]):
                for j in range(value_grid.shape[1]):
                    state_ = observation(vwc.coord(i,j), state[1], state[2])
                    value_grid[i,j] = qlearn.state_value(state_)


import pyworld.toolkit.tools.fileutils as fu
import copy    
 
def train(white_mind, green_mind=None, orange_mind=None, cont=False, steps=1000):
    minds = {}
    minds[vwc.colour.white], minds[vwc.colour.green], minds[vwc.colour.orange] = vwutils.process_minds(white_mind, green_mind, orange_mind)
    
    if not cont:
        qlearn = TabularQLearning(action_space)
    else:
        qlearn = fu.load("test.pickle")
    
    
    grid = saveload.load('test.vw')
    
    for i in range(1, steps):
        
        if not i % 100:
            fu.save("test.pickle", qlearn)
        
        env = vwenv.GridEnvironment(vwenv.GridAmbient(copy.deepcopy(grid), minds))
        list(env.ambient.agents.values())[0].mind.surrogate.qlearn = qlearn #sigh... avoid a large copy
        
        #print(env.ambient.agents)
        print("episode:", i, "dirts: ", len(env.ambient.grid._get_dirts()))
        #for i in range(2):
        while len(env.ambient.grid._get_dirts()) != 0:
            env.evolveEnvironment()
        
        env.evolveEnvironment() #1 more time to allow for the final update
            
    fu.save("test.pickle", qlearn)

            

def tkcolour(r,g,b):
    return "#%02x%02x%02x" % (r, g, b)

def redraw_hook(root):
    
    #print(globals()['main_interface'])
    main_interface = None
    for widg in root.winfo_children():
        if isinstance(widg, vwv.VWInterface):
            main_interface = widg
            break
    if main_interface is None:
        print("ANIMATION SET UP FAILED - COULD NOT FIND MAIN INTERFACE")
        return
 
    def redraw(self):
        old_redraw(self)
        for rect in self._q_rectangles:
            self.canvas.delete(rect)
        
        global value_grid
        if len(np.unique(value_grid)) != 1:
            n_values = (du.normalise(value_grid) * 255).astype(int)
        else:
            n_values = value_grid.astype(int)
        
        inc = vwv.GRID_SIZE / self.grid.dim
        for i in range(self.grid.dim):
            for j in range(self.grid.dim):
                rect = self.canvas.create_rectangle(i * inc,j * inc,
                                                    (i + 1) * inc,(j + 1) * inc, fill=tkcolour(255-n_values[i,j],n_values[i,j],0))
                self._q_rectangles.append(rect)
                self.canvas.tag_lower(rect)
    
        #print("do some other things...")
        
    old_redraw = vwv.VWInterface._redraw
    vwv.VWInterface._redraw = redraw 
    main_interface._q_rectangles = []
    
    global value_grid
    value_grid = np.zeros((main_interface.grid.dim, main_interface.grid.dim))
    root.after(0, main_interface._redraw)

    #it is 1 step to late it seems!
    
#mind = QLearningMind(action_space, t=0.1) 
#train(mind)

#grid = saveload.load('test.vw')
qlearn=fu.load("test.pickle")
mind = QLearningMind(action_space, qlearn=qlearn, t = 0.1, train=False)  

#mind = QLearningMind(action_space, t = 0.1, train=False) 
vacuumworld.run(mind, skip=True, load='test.vw', speed=0.6, tkhooks=[redraw_hook])





