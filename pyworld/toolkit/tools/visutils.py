#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:41:32 2019

@author: ben
"""
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

import numpy as np

from enum import Enum

from PIL import Image, ImageDraw, ImageFont
import os



from . import fileutils as fu
from . import datautils as du

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
    
    '''
    def interactive(self, grid):
        self.grid = grid
        plt.ion()
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.ax = self.fig.add_subplot(111)
        self.data = self.ax.imshow(self.grid)
        #(self.grid)
        
    def interactive_update(self, *args):
        print("update...")
        if not self.stop_interactive:
            self.grid = self.update(self.grid, *args)
            self.ax.clear()
            
            self.data = self.ax.imshow(self.grid)
            plt.draw()
            #plt.pause(0.001)
            #show(self.grid)
    ''' 
    
    
    def handle_close(self, evt):
        plt.close(self.fig)
        self.stop_interactive = True


class track2D:
    
    def __init__(self, figure, x, z):
        assert z.shape[1] == 2 #2d plot!
        self.figure = figure
        self.x = x
        self.z = z
        self.p = np.array([np.inf, np.inf])
        self.figure.canvas.mpl_connect('motion_notify_event', lambda e: self.__imagetrack(e))
        
    def __imagetrack(self, event):
        if event.xdata is not None and event.ydata is not None:
            p = np.array([event.xdata, event.ydata])
            zi = np.argmin(np.sum((self.z - p)**2, axis=1)) #TODO a bit inefficient...
            cv2.imshow('tracking', self.x[zi])

def umap(x, y=None, **kwargs):
    import umap
    if y is not None:
        assert len(y.shape) == 1
        
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0],-1)
    assert len(x.shape) == 2
    
    reducer = umap.UMAP(**kwargs)
    reducer.fit(x, y)
    
    embedding = reducer.transform(x)
    
    fig = __new_plot(None, draw=True, clf=True)
    if y is not None:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
        u = np.unique(y)
        plt.colorbar(boundaries=np.arange(len(u) + 1)-0.5).set_ticks(u)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
    plt.gca().set_aspect('equal', 'datalim')
    
    return embedding, fig


def matplot_close(fig='all'):
    plt.close(fig)

def matplot_activate(use='Qt5Agg'):
    matplotlib.use(use)
    
def matplot_deactivate():
    matplotlib.use('Agg')

def matplot_isopen():
    return plt.get_fignums()

def matplot_isclosed():
    return not plt.get_fignums()

class plt_events(Enum):
    press = 'button_press_event' 
    release = 'button_release_event' 
    draw = 'draw_event' 	
    key_press = 'key_press_event' 	
    key_release = 'key_release_event'
    motion = 'motion_notify_event' 
    pick = 'pick_event' 
    resize = 'resize_event' 	
    scroll = 'scroll_event' 
    enter = 'figure_enter_event' 	
    exit = 'figure_leave_event' 
    enter_axes = 'axes_enter_event' 	
    exit_axes = 'axes_leave_event'

def listen(figure, event, listener):
    print(event)
    figure.canvas.mpl_connect(event.value, listener)

def savefig(fig, path, extension = ".png"):
    if not fu.has_extension(path):
        path = path + extension
    path = fu.file(path)
    
    img = figtoimage(fig)

    plt.imsave(path, img)

def figtoimage(fig):
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape((h, w, 3))
    
def savevideo(iterator, path, extension = ".mp4", fps=30):
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(iterator, fps=fps)
    
    if not fu.has_extension(path):
        path = path + extension
    path = fu.file(path)
    
    clip.write_videofile(path) # default codec: 'libx264', 24 fps
    
def save(image, path, extension = ".png"):
    image = du.normalise(image)
    if not fu.has_extension(path):
        path = path + extension
    path = fu.file(path)
    print(path)
    plt.imsave(path, colour(image))

def resize_all(images, shape):
    result = np.empty((images.shape[0], *shape, images.shape[3]))
    for i in range(len(images)):
        result[i] = cv2.resize(images[i], shape)[:,:,np.newaxis]
    return result

def crop(image, shape):
    cv2.crop

def resize(image, shape, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=interpolation)

def scale(image, x, y):
    return cv2.resize(image, None, fx=x, fy=y)

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def colour(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def CHW(data): #TORCH FORMAT
    if len(data.shape) == 2:
        return data[np.newaxis,:,:]
    elif len(data.shape) == 3:    
        return data.transpose((2,0,1))
    elif len(data.shape) == 4:
        return data.transpose((0,3,1,2))
    else:
        raise ValueError("invalid dimension: " + str(len(data.shape)))
    
def HWC(data): #CV2 FORMAT
    if len(data.shape) == 2:
        return data[:,:,np.newaxis]
    if len(data.shape) == 3:    
        return data.transpose((1,2,0))
    elif len(data.shape) == 4:
        return data.transpose((0,2,3,1))
    else:
        raise ValueError("invalid dimension: " + str(len(data.shape)))
        
        
def channels_to_torch(data):
    print("DEPRECATED USE CHW")
    if len(data.shape) == 4:
        return data.reshape((data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
    elif len(data.shape) == 3:
        return data.reshape((data.shape[2], data.shape[0], data.shape[1]))

def channels_to_cv(data):
    print("DEPRECATED USE HWC")
    if len(data.shape) == 4:
        return data.reshape((data.shape[0], data.shape[2], data.shape[3], data.shape[1]))
    elif len(data.shape) == 3:
        return data.reshape((data.shape[1], data.shape[2], data.shape[0]))
    
def shape_to_torch(shape):
    if len(shape) == 4:
        return (shape[0], shape[3], shape[1], shape[2])
    elif len(shape) == 3:
        return (shape[2], shape[0], shape[1])

def shape_to_cv(shape):
    if len(shape) == 4:
        return (shape[0], shape[2], shape[3], shape[1])
    elif len(shape) == 3:
        return (shape[1], shape[2], shape[0])

'''
def gallery(array, ncols=3):
    nindex, height, width = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols))
    return result
'''
def label_colours(labels, alpha=0.8):
    colours = cm.rainbow(np.linspace(0,1,len(labels)))
    colours[:,3] *= alpha
    result = {}
    for i in range(colours.shape[0]):
        result[labels[i]] = colours[i]
    return result

def __new_plot(fig, draw=True, clf=True):
    if draw:
        matplot_activate()
        plt.ion()
    else:
        matplot_deactivate()
        plt.ioff()
        
    if fig is None:
        fig = plt.figure()
        
    if clf:
        fig.clf()
    
    plt.figure(fig.number)
    
    return fig

def __update_plot(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, draw=True, pause=0.001):
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if draw:
        plt.draw()
        plt.pause(pause)

def histogram2D(x, bins, fig=None, draw=True, clf=True, title=None, stacked=False, xlabel=None, ylabel=None, labels=None, colour=None, alpha=1., log=False):
    fig = __new_plot(fig, draw=draw, clf=clf)    
    plt.hist(x, bins, color=colour, alpha=alpha, log=log, label=labels, stacked=stacked)
    if labels:
        plt.legend()
    __update_plot(title, xlabel, ylabel)
    return fig

def plot2D(model, x, y=None, fig=None, clf=True, marker=".", colour=None, alpha=0.8, 
                  title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, pause=0.001, draw = True):
    fig = __new_plot(fig, draw=draw, clf=clf)
    
    z = du.collect(model, x)
    
    assert z.shape[1] == 2 #...hmmmm
    
    if y is None:
        __plot2D(z, marker=marker, colour=colour)
    else:
        __plot2D_split(z, y, colours=colour, alpha=alpha)
    __update_plot(title, xlabel, ylabel, xlim, ylim, draw, pause)
   
    return fig

def __plot2D(x, marker = '.', colour = None, alpha=1.):
    plt.plot(x[:,0], x[:,1], marker, color=colour, alpha=alpha)

def __plot2D_split(x, y, colours=None, alpha=1.):
    line_handles = []
    label_handles = []

    data = du.splitbylabel(x, y)
    
    if colours is None:
        colours = label_colours(list(data.keys()), alpha=alpha)
    else:
        assert(len(colours) == len(data.keys()))
    
    for label, d in data.items():
        line = plt.scatter(d[:,0], d[:,1], color=colours[label],  edgecolors='none', label=label, marker='.')
            
        line_handles.append(line)
        label_handles.append(label)
        
    plt.legend(line_handles, label_handles, loc="upper right")



#123, 252, 3
def show_attention(values, queries, weights, embed_size=None, display_shape=(240,480),
                   wait=240, line_colour=(0.48235, 0.98823, 0.01176), line_tickness=2):
    if embed_size is None:
        embed_size = np.max([*values.shape, *queries.shape])
    embed = np.ones((2 + embed_size, embed_size * 2 + 3, values.shape[2]))
    embed[1:queries.shape[0]+1,1:queries.shape[1]+1] = queries
    embed[1:values.shape[0]+1,2+embed_size:2+embed_size+values.shape[1]] = values
    
    values_size = np.prod(values.shape[:-1])
    queries_size = np.prod(queries.shape[:-1])
    
    lines1, lines2 = [], []
    incx = int(display_shape[1] / embed.shape[1])
    incy = int(display_shape[0] / embed.shape[0])
    incx2 = int(incx / 2)
    incy2 = int(incy / 2)
    for i in range(queries_size):
        for j in range(values_size):
            lines1.append((incx * ((i % queries.shape[0]) + 1) + incx2, incx * ((j % values.shape[0]) + 2 + embed_size) + incx2))
            lines2.append((incy * ((i // queries.shape[1]) + 1)+ incy2, incy * (( j // values.shape[1]) + 1) + + incy2))

    
    embed = resize(embed.astype(np.float32), display_shape)
    embed = colour(embed)
    
    #construct video

    ii = 0
    frames = []  
    plt.ioff()
    for i in range(values_size, queries_size * values_size + 1, values_size):
        plt.clf()
        plt.imshow(embed)
        for p1, p2 in zip(lines1[ii:i], lines2[ii:i]):
            print(p1, p2)
            plt.plot(p1, p2, linewidth=line_tickness, alpha=0.5, color=line_colour)
            #cv2.line(change, p1, p2, line_colour, line_tickness, lineType=cv2.LINE_AA)
        frames.append(figure_to_numpy(plt.gcf()))
        ii = i

    play(frames, 'attention', wait=wait, repeat=True)

def covariance(images, title=None, ):
    print(images.shape)
    flattened = images.reshape(-1, np.prod(images[0].shape)).T
    covar = np.cov(flattened)
    print(covar.shape)
    plt.figure()
    plt.title(title)
    #plt.axis('off')
    plt.imshow(covar, cmap='viridis')
    
    
def figure_to_numpy(fig):
    # draw the renderer
    fig.canvas.draw() 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height() 
    return np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h,w,3)  
    
def gallery(array, ncols=3):
    array = __HWC_format(array) 
    nindex, height, width, intensity = array.shape    
    nrows = nindex//ncols

    fill = abs(nindex - ((nrows + 1) * ncols))
    #print(ncols, nrows, fill, width, height)
    if fill:
        zeros = np.zeros((fill, height, width, intensity))
        #print(array.shape, zeros.shape)
        array = np.concatenate((array, zeros))
        nindex, height, width, intensity = array.shape    
        nrows = nindex//ncols
        
    #assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def __HWC_format(array):
    '''
        dim(array) == 2 transform to HWC format
        dim(array) == 3 transform to HWC format
        dim(array) == 4 transform to BHWC format
    '''
    chw = 0
    hwc = 2
    if len(array.shape) == 3:
        pass
    elif len(array.shape) == 2:
        return array[:,:,np.newaxis]
    elif len(array.shape) == 4:
        chw += 1
        hwc += 1
    else:
        raise ValueError("invalid image dimension: " + str(len(array.shape)))
      
    if array.shape[hwc] == 1 or array.shape[hwc] == 3 or array.shape[hwc] == 4:
        return array
    elif array.shape[chw] == 1 or array.shape[chw] == 3 or array.shape[chw] == 4:
        return HWC(array)    

def __HWC_show(name, array):
    array = __HWC_format(array)
    cv2.imshow(name, array)

def show(array, name='image', wait=60):
    __HWC_show(name, array)
    print("SHOW")
    if wait >= 0:
        return cv2.waitKey(wait) == ord('q')
    return

def play(video, name='video', wait=60, repeat=False):
    while True: 
        for f in video:
            __HWC_show(name, f)
            if cv2.waitKey(wait) == ord('q'):
                close(name)
                return
        if not repeat:
            close(name)
            return
    

def close(name=None):
    if name is None: 
        cv2.destroyAllWindows()
    else:
        cv2.destroyWindow(name)
        
def waitclose(wait=60):
    if cv2.waitKey(wait) == ord('q'):
        close()
        return True
    return False

class __images:
    
    def character(self, char):
        W = H = 14
        img = Image.new('RGB', (W,H), color = (0,0,0)) #(0,0,0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(os.path.dirname(__file__) + "/ArialCE.ttf", 18)

        w,h = font.getsize(char)
        draw.text(((W-w)/2,(H-h)/2 - 2), char, font=font, fill=(255,255,255)) #dont ask...        
        return CHW(gray(np.array(img)) / 255.)




images = __images()