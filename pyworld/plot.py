#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:16:54 2019

@author: ben
"""
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import tensorflow as tf
import collections

from scipy import stats
from scipy import ndimage
import numpy as np

def load_tensor(path, include_time=False):
    arr = []
    g = gen_csv(path)
    headers = next(g)
    for row in g:
        arr.append(row)
    tensor = np.array(arr, copy=False)
    return tensor[:,0 if include_time else 1:], headers   
    
def gen_csv(path):
    f = open(path, 'r')
    r = csv.reader(f)
    for row in r:
        yield row
    f.close()
    
def smooth(y, box_pts):
    y = np.concatenate((np.array([np.mean(y[:box_pts])]*box_pts), y, np.array([np.mean(y[-box_pts:])]*box_pts))) #avoid edge artefacts
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth[box_pts:-box_pts]

def smooth2(y, sigma):
    return ndimage.gaussian_filter1d(y, sigma)
    
def pad_y(y, smth):
    print(y, smth)
    print(np.array([y[0]]*smth))
    print(np.array([y[-1]]*smth))
    
    return y

def get_tensors(path):
    out = collections.defaultdict(list)
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:
            out[v.tag].append((e.step, v.simple_value))
    for k,v in out.items():
        out[k] = np.array(v)
    return out

def plot_tensors(path, smth=0, step=1, name=None, title=None, legend=None, zthresh=None):
    if isinstance(path, str):
        path = [path]
    outs = dict()
    for p in path:
        outs[p] = get_tensors(p)

    if name == None:
        name = list(reduce(lambda x, y: x.union(outs[y].keys()), path, set()))
    
    print(list(reduce(lambda x, y: x.union(outs[y].keys()), path, set())))
    
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=20)
    if zthresh == None:
        zthresh = [3.] * len(name)
    if isinstance(step, int):
        step = [step]*len(name)
    if isinstance(smth, int):
        smth = [smth]*len(name)   

    axis = dict()
    for i in range(len(name)):
        axis[name[i]] = fig.add_subplot(1,len(name), i+1)
        axis[name[i]].set_ylabel(format_label(name[i]), fontsize=16)
        axis[name[i]].set_xlabel('steps', fontsize=16)
    lines = dict()
    for j in range(len(path)):
        p = path[j]
        for i in range(len(name)):
            k = name[i]
            line = _plot_tensor(axis[k], k, outs[p][k], smth[i], step[i], zthresh=zthresh[i])
        if legend:
            lines[legend[j]] = line
    if legend:
        plt.figlegend(tuple(lines.values()), tuple(lines.keys()), 'best')
    return fig
    

def _plot_tensor(ax, name, tensor, smth=0, step=1, zthresh=3.):
    #fig = plt.figure()
    #fig.suptitle(name, fontsize=20)
    if tensor.shape[0] / 10 > step:
        index = np.arange(0,tensor.shape[0]) % step == 0
    else:
        index = np.arange(0,tensor.shape[0])
    x = tensor[index,0]
    y = tensor[index,1]
    first = len(ax.get_lines()) == 0
    if smth > 0:
        ys = mask_outs(y, zthresh)
        ys = smooth2(ys, smth)
        line, = ax.plot(x, ys)
        color = line.get_color()
        color = list(matplotlib.colors.to_rgba(color))
        color[3] = 0.2
        ax.plot(x, y, color=tuple(color))
        axis_limits(ax, x, ys, first)
    else:
        line, = ax.plot(x, y)
        axis_limits(ax, x,y, first)
    ax.ticklabel_format(axis='x', scilimits=(0,0))
    
    return line

def format_label(label):
    return label.split("/")[-1]    
    
def mask_outs(y, thresh=3):
     #remove outliers
    z = np.abs(stats.zscore(y))
    y = np.copy(y)
    y[np.where(z >= thresh)] = np.mean(y)
    return y
    

def axis_limits(ax, x, y, first=False):
    xmin = min(x)
    xmax = max(x)
    xrange = (xmax - xmin)/16.
    ymin = min(y)
    ymax = max(y)
    yrange = (ymax - ymin)/16.
    if not first:
        ax.set_ylim([min(ymin - yrange, ax.get_ylim()[0]), max(ymax + yrange, ax.get_ylim()[1])])
        ax.set_xlim([min(xmin - xrange, ax.get_xlim()[0]), max(xmax + xrange, ax.get_xlim()[1])])
    else:
        
        ax.set_ylim([ymin - yrange, ymax + yrange])
        ax.set_xlim([xmin - xrange, xmax + xrange])
        


if __name__ == "__main__":
    '''
    DDQN_path = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/DDQN-Target-195/events.out.tfevents.1556303991.Ben-Recoil-II'
    DQN_path = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/DQN-Target-205/events.out.tfevents.1556299524.Ben-Recoil-II'
    fig = plot_tensors([DQN_path,DDQN_path], legend=['DQN', 'DDQN'], smth=[10, 10, 2], step=[10,5,1], name=['qval', 'loss', 'reward'], title='DQN vs DDQN')

    DQN_path = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/DQN-Target-205/events.out.tfevents.1556299524.Ben-Recoil-II'
    VS_path = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/DQN-Vanilla-success/events.out.tfevents.1557407508.Ben-Recoil-II'
    fig = plot_tensors([DQN_path,VS_path], legend=['DQN', 'Vanilla'], smth=[10, 10, 5], zthresh=[3,2,3], step=[10,20,1], name=['qval', 'loss', 'reward'], title='Vanilla vs Target')
    '''  
    '''
    VS_path = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/DQN-Vanilla-success/events.out.tfevents.1557407508.Ben-Recoil-II'
    VF_path = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/DQN-Vanilla/events.out.tfevents.1556306536.Ben-Recoil-II'
    fig = plot_tensors([VF_path,VS_path], legend=['Fail', 'Success'], smth=[10, 10, 5], zthresh=[3,2,3], step=[10,20,1], name=['qval', 'loss', 'grad/grad_max'], title='Vanilla Problems')
    '''
    '''
    D3QN_106o4 = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/D3QN-Target/events.out.tfevents.1557413572.Ben-Recoil-II'
    DQN_105 = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/D3QN-epsilon10t5/events.out.tfevents.1557482308.Ben-Recoil-II'
    DQN_105m2 = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/D3QN-epsilon10t5m2/events.out.tfevents.1557487309.Ben-Recoil-II'
    DQN_105m2_2 = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/D3QN-epsilon10t5m2-2/events.out.tfevents.1557490034.Ben-Recoil-II'
    fig = plot_tensors([D3QN_106o4,DQN_105m2, DQN_105m2_2, DQN_105], legend=['10^6/4', '10^5/2 (1)', '10^5/2 (2)', '10^5'], smth=[0, 1, 5], zthresh=[100,10,10], step=[10,1,10], name=['epsilon', 'reward', 'loss'], title='D3QN Epsilon Experiments')
    '''
    
    DQN_105 = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/D3QN-epsilon10t5/events.out.tfevents.1557482308.Ben-Recoil-II'
    NDQN = '/home/ben/Documents/repos/ExperimentsWithRL/DeepQ/runs/keep/NDQN-epsilon-low/events.out.tfevents.1556313776.Ben-Recoil-II'
    fig = plot_tensors([DQN_105,NDQN], legend=['D3QN', 'NDQN'], smth=[10, 10, 5], zthresh=[3,2,3], step=[10,20,1], name=['qval', 'loss', 'reward'], title='D3QN vs NDQN')


'''    
#t, heads = load_tensor("/home/ben/Downloads/run-keep_DQN-Vanilla-tag-reward.csv")
#t, heads = load_tensor("/home/ben/Downloads/run-Apr26_23-13-49_Ben-Recoil-II-tag-reward.csv")



x = t[:,0]
y = t[:,1]

print(x)
print(y)

plt.plot(x,y)
'''