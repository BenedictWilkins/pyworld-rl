
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

import numpy as np

class HeatmapAnimation:
    
    def __init__(self, update, interval=10):
        self.grid = None
        self.update = update
        self.interval = interval
        self.fig = None
        #self.stop_interactive = False
        
    # function to update figure
    def __update__(self, *args):
        print("update", args)
        # set the data in the axesimage object
        self.grid = self.update(self.grid)
        self.im.set_array(self.grid)
        # return the artists set
        return self.im,
    
    def show(self, grid, vmin=0., vmax=1.):
        self.grid = grid
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.im = plt.imshow(grid, vmin=vmin, vmax=vmax)
        
        self.ani = FuncAnimation(self.fig, self.__update__, interval=self.interval)
        plt.show()
    
    def handle_close(self, evt):
        plt.close(self.fig)
        self.stop_interactive = True
