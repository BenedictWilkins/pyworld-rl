#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 14-06-2019 11:41:32

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

import numpy as np

from enum import Enum

from PIL import Image, ImageDraw, ImageFont
import os

from .. import fileutils as fu
from .. import datautils as du
from .. import torchutils as tu

from . import colour
from . import transform
from . import animation
from . import detection
from . import plot # plotly
from . import jupyter #IPython visuals that only work well in jupyter...


try: #TODO move somewhere
    import moviepy.editor as mpy
except:
    mpy = None

__all__ = ('transform', 'animation', 'detection', 'plot')


def hstitch(images):
    pass

def vstitch(images):
    pass


def grid(domain, n=100):
    """ Create a 2D grid of n^2 points over the domain. 

    Args:
        domain (int, float, tuple): TODO
        n (int, optional): number of points along each axis. Defaults to 100.

    Raises:
        NotImplementedError: tuple domain... TODO

    Returns:
        np.ndarray : points in the grid, shape n^2 x 2
    """
    if isinstance(domain, (int, float)):
        domain = ((-domain/2, domain/2), (-domain/2, domain/2))
    else:
        pass # raise NotImplementedError("TODO")

    z1, z2 = np.meshgrid(np.linspace(domain[0][0],domain[0][1],n), np.linspace(domain[1][0],domain[1][1],n))
    z1, z2 = z1.flatten()[:,np.newaxis], z2.flatten()[:,np.newaxis]
    return np.concatenate((z1, z2), axis=1)      

def savevideo(iterator, path, extension = ".mp4", fps=30):
    if mpy is not None:
        sequence = [x for x in iterator] #sigh...
        clip = mpy.ImageSequenceClip(sequence, fps=fps)
        
        if not fu.has_extension(path):
            path = path + extension
        path = fu.file(path)
        clip.write_gif(path)

        #clip.write_videofile(path) # default codec: 'libx264', 24 fps
    else:
        raise ImportError('saving a video requires the module: \'moviepy\'.')

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
    return bool(plt.get_fignums())

def matplot_isclosed():
    return not matplot_isopen()

def cv_isopen(name):
    try:
        return cv2.getWindowProperty(name, 0) != -1
    except:
        return False

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
    return np.flip(buf.reshape((h, w, 3)), 2) #bgr format for opencv!
    
def save(image, path):
    image = du.normalise(image)
    if not fu.has_extension(path):
        path = path + ".png"
    path = fu.file(path)
    print(path)
    plt.imsave(path, transform.colour(image))

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

def histogram2D(x, bins, **kwargs): #for legacy reasons.. TODO remove
    return histogram(x, bins, **kwargs)

def histogram(x, bins, fig=None, draw=True, clf=True, title=None, stacked=False, xlabel=None, ylabel=None, labels=None, colour=None, alpha=1., log=False):
    '''
        Creates a histogram of the given data.
        Arguments:
            x: 1D array (or collection o)
            bins: number of bins to use
            # TODO
    '''
    fig = __new_plot(fig, draw=draw, clf=clf)    
    plt.hist(x, bins, color=colour, alpha=alpha, log=log, label=labels, stacked=stacked)
    if labels:
        plt.legend()
    __update_plot(title, xlabel, ylabel)
    return fig

def plot2D(model, x, y=None, fig=None, clf=True, marker=".", colour=None, alpha=0.8, 
                  title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, pause=0.001, draw = True):
    fig = __new_plot(fig, draw=draw, clf=clf)
    
    z = tu.collect(model, x)
    
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

    
    embed = transform.resize(embed.astype(np.float32), display_shape)
    embed = transform.colour(embed)
    
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

def covariance(images, title=None): #??
    print(images.shape)
    flattened = images.reshape(-1, np.prod(images[0].shape)).T
    covar = np.cov(flattened)
    print(covar.shape)
    plt.figure()
    plt.title(title)
    #plt.axis('off')
    plt.imshow(covar, cmap='viridis')
    
    
def figure_to_image(fig):
    '''
        Converts a figure into an image.
        Arguments:
            fig: the figure to convert
    '''
    # draw the renderer
    fig.canvas.draw() 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height() 
    return np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h,w,3)  
    
def gallery(images, cols=3):
    '''
        Creates a 2D gallery of images from a 1D array of images with a given number of columns.
        Arguments:
            images: an array of N images in HWC or CHW format
            cols: number of columns in the image gallery
    '''
    array = __HWC_format(images) 
    if len(array.shape) == 3:
        array = array[:,:,:,np.newaxis] #NHWC format
        

    nindex, height, width, intensity = array.shape    
    nrows = int(np.ceil(nindex/cols))

    fill = abs(nindex - ((nrows) * cols))
    #print(ncols, nrows, fill, width, height)
    if fill:
        zeros = np.zeros((fill, height, width, intensity))
        #print(array.shape, zeros.shape)
        array = np.concatenate((array, zeros))
        nindex, height, width, intensity = array.shape    
        nrows = nindex//cols
        
    #assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, cols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*cols, intensity))
    return result

def hgallery(x, n=10):
    if transform.isCHW(x):
        if not transform.isHWC(x):
            x = transform.HWC(x)

    if n is None:
        n = x.shape[0]
    m,h,w,c = x.shape
    n = min(m, n) #if n is larger, just use m
    if m % n != 0:
        pad = ((0, n - (m % n)),*([(0,0)]*(len(x.shape)-1)))
        x = np.pad(x, pad)
        m,h,w,c = x.shape
        
    return x.swapaxes(1,2).reshape(m//n, w * n, h, c).swapaxes(1,2)





def show(image, name='image'):
    '''
        Shows the given image. See also wait() and close()
    '''
    __HWC_show(name, image)


def play(video, name='video', wait=30, repeat=False, key='q'): #TODO fix repeat... (it relies on the iterator)
    '''
        Plays a video (a sequence or iterable of images).
        Arguments:
            video: an interable of images (frames)
            name: name of the display window, default 'video'
            wait: time to wait between each frame (ms), default 60
            repeat: whether to repeat the video once the iterable has finished, default False
            key: to press to close the video
    '''
    while True: 
        for f in video:
            __HWC_show(name, f)
            if cv2.waitKey(wait) == ord(key):
                close(name)
                return
        if not repeat:
            close(name)
            return
        

def close(name=None):
    '''
        Closes the named window (or all windows if name is None).
        Arguments:
            name: of the window to close, default None
    '''
    if name is None: 
        cv2.destroyAllWindows()
    else:
        cv2.destroyWindow(name)

def wait(name=None, key='q'):
    '''
        Waits for the key (defualt 'q') to be pressed and then exits the 
        named diplay window (or all open windows if name is None).
        Arguments:
            name: of the window to close, default None
            key: to press to close the window(s)
    '''
    while cv2.waitKey(120) != ord(key):
        pass
    close(name)

# -----------------  -----------------  ----------------
# -----------------  ----- USEFUL ----  ----------------
# -----------------  -----------------  ----------------

    




def __HWC_format(array): #transform to HWC format
    '''
        dim(array) == 2 transform to HWC format
        dim(array) == 3 transform to HWC format
        dim(array) == 4 transform to BHWC format
    '''

    chw = 0
    hwc = 2
    if len(array.shape) == 3:
        return array
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
        return transform.HWC(array)    

def __HWC_show(name, array, size=None): #show with HWC transform
    array = __HWC_format(array)
    cv2.imshow(name, array)
    cv2.waitKey(60) #... so weird
