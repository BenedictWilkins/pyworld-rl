#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d

class SurfaceAnimation:
    
    def __init__(self, update, X, Y, Z=None, interval=200):
        self.interval = interval
        self.X,self.Y = np.meshgrid(X,Y)
        if not Z:
            self.Z = np.zeros((self.X.shape[0],self.Y.shape[0]))
        self.update = update
        
    def __update__(self, i):
        self.X,self.Y,self.Z = self.update(i, self.X,self.Y,self.Z)
        self.ax.clear()
        self.ax.plot_surface(self.X,self.Y,self.Z, cmap='viridis')
        
    def show(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.plot_surface(self.X,self.Y,self.Z,cmap='viridis')
        self.ani = FuncAnimation(self.fig, self.__update__, interval=self.interval)
        plt.show()


class HeatmapAnimation:
    
    def __init__(self, grid, update, interval=200):
        self.grid = grid
        self.update = update
        self.interval = interval
        self.data = None
        self.stop_interactive = False
        
    def __update__(self, i):
        self.grid = self.update(i, self.grid)
        self.ax.clear()
        self.data.set_data(self.grid)
    
    def show(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ani = FuncAnimation(self.fig, self.__update__, interval=self.interval)
        plt.show()
        
    def interactive(self):
        plt.ion()
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.ax = self.fig.add_subplot(111)
        self.data = self.ax.imshow(self.grid)
        
    def interactive_update(self, i):
        if not self.stop_interactive:
            self.grid = self.update(i, self.grid)
            self.ax.clear()
            self.ax.imshow(self.grid)
            plt.draw()
            plt.pause(0.001)
            
    def handle_close(self, evt):
        plt.close(self.fig)
        self.stop_interactive = True
        
if __name__ == "__main__":
    def update(i, grid):
        return np.random.random((10,10))
    
    h = HeatmapAnimation(np.random.random((10,10)), update)
    h.interactive()
    
    for i in range(10000):
        h.interactive_update(0)
    
    '''
    def update(i,X,Y,Z):
        print("update animation")
        return X,Y, Z+0.1
    l = np.linspace(0,1,20)
    s = SurfaceAnimation(update,l,l)
'''