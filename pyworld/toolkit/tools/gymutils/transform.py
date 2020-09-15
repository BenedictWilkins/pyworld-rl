#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-03-13 12:06:32

author: Benedict Wilkins
"""

import gym
import numpy as np
import cv2
from collections import namedtuple
#from enum import Enum

from ..visutils import transform as T

def stack(states, frames=3, copy=True):
    """ Stack states. with an input of states [s1, s2, ..., sn] where each state is a grayscale image, the output will is given as:


    Args:
        states (numpy.ndarray): states [s1, s2, ..., sn] as images NHWC format.
        frames (int, optional): number of states to stack. Defaults to 3.
        copy (bool, optional): create a new array to store the ouput. WARNING: without a copy, modifying the array can have undesirable effects as stride_tricks is used. Defaults to True.

    Returns:
        numpy.ndarray: stacked states
    """
    states = states.squeeze() #remove channels
    shape = states.shape
    assert len(shape) == 3 # NHW format

    # stack frames
    stacked = np.lib.stride_tricks.as_strided(states, shape=(shape[0] - frames, *shape[1:], frames), strides=(*states.strides, states.strides[0]))
    if copy:
        stacked = np.copy(stacked)
    return stacked





def maxpool(state, factor=2): #TODO...
    input_size = 128
    output_size = 64
    bin_size = input_size // output_size
    small_image = large_image.reshape((output_size, bin_size, 
                                   output_size, bin_size, 3)).max(3).max(1)

def atari(states):
    """ Transformation specific to the Atari suit of games provided by the Arcade Learning Environment (ALE). The states is assumed to be in NHWC format.

    Args:
        states (np.ndarray): a collection of states from an Atari game, NHWC format

    Returns:
        np.ndarray: transformed states
    """
    states = T.grey(states, components=(0.299, 0.587, 0.114))
    states = T.resize(states, 84, 110)
    states = T.crop_all(states, ysize=(18, 102)) #TODO swap resize/crop for efficiency.
    return states


if __name__ == "__main__":
    print("test")
    t = np.random.randint(0, 10, size=(5, 1, 3, 3))
    s = stack(t)
    print(s)


def assert_box(space):
    assert isinstance(space, gym.spaces.Box)


def assert_interval(space, high=255, low=0):
    u = np.unique(space.high.ravel())
    assert len(u) == 1
    assert u[0] == high
    u = np.unique(space.low.ravel())
    assert len(u) == 1
    assert u[0] == low


def assert_unique(space):
    high = np.unique(space.high.ravel())
    assert len(high) == 1
    low = np.unique(space.low.ravel())
    assert len(low) == 1
    return low[0], high[0]


class __OM_Stack:

    def __init__(self, env, stack=3):
        assert stack >= 1
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        channels = env.observation_space.shape[0]
        assert channels == 1  # must be in CHW format!

        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(
            env.observation_space.shape[0] * stack, *env.observation_space.shape[1:]), dtype=env.observation_space.dtype)

        self.states = np.zeros(
            (stack*channels, *env.observation_space.shape[1:]), dtype=self.observation_space.dtype)
        self.indx = (np.arange(0, stack) - 1) % stack

    def __call__(self, state):
        self.states = self.states[self.indx]  # this will make a copy
        self.states[0] = state[0]
        return self.states


class __OM_Gray:

    def __init__(self, env):
        assert_box(env.observation_space)
        assert_interval(env.observation_space)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(
            *env.observation_space.shape[:2], 1), dtype=np.float32)

    def __call__(self, state, *args):
        state = (state[:, :, 0] * 0.299 + state[:, :, 1] *
                 0.587 + state[:, :, 2] * 0.114).astype(np.float) / 255.
        return state.reshape(*state.shape, 1)


class __OM_Interval:

    def __init__(self, env):
        assert_box(env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)

    def __call__(self, state, *args):
        state.astype(np.float) / 255.0


class __OM_Crop:

    def __init__(self, env, box):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        assert len(box) == 4
        self.box = box
        w = box[2] - box[0]
        h = box[3] - box[1]
        c = env.observation_space.shape[2]  # assume HWC format
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(h, w, c), dtype=env.observation_space.dtype)

    def __call__(self, state):
        return state[self.box[1]:self.box[3], self.box[0]:self.box[2], :]


class __OM_Resize:

    def __init__(self, env, shape):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=shape, dtype=np.float32)

    def __call__(self, state):
        return cv2.resize(state, self.observation_space.shape, interpolation=cv2.INTER_AREA)


class __OM_Binary:
    def __init__(self, env, threshold=0.5):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)

        self.threshold = threshold * (high - low)
        self.observation_space = gym.spaces.Box(
            low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)

    def __call__(self, state):
        return np.where(state > self.threshold, 1., 0.)


class __OM_Default:

    def __init__(self, env):
        assert_box(env.observation_space)
        assert_interval(env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=0., high=1.0, shape=(84, 84, 1), dtype=np.float32)

    def __call__(self, state):
        img = (state[:, :, 0] * 0.299 + state[:, :, 1]
               * 0.587 + state[:, :, 2] * 0.114)  # gray
        img = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)  # resize
        img = img[18:102, :] / 255.0  # crop
        return img.reshape((*img.shape, 1)).astype(np.float32)


class __OM_CHW:

    '''
        Transforms states from HWC to CHW format suitable for pytorch image processing. 
    '''

    def __init__(self, env):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        # assumes HWC format
        channels = env.observation_space.shape[2]
        height = env.observation_space.shape[0]
        width = env.observation_space.shape[1]

        assert channels == 1 or channels == 3 or channels == 4
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(
            channels, height, width), dtype=env.observation_space.dtype)

    def __call__(self, state):
        return state.transpose((2, 0, 1))
        # return np.swapaxes(state, 0, 2)


class __OM_HWC:

    '''
        Transforms states from CHW to HWC format suitable for opencv (or PIL) processing.
    '''

    def __init__(self, env):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        # assumes CHW format
        channels = env.observation_space.shape[0]
        height = env.observation_space.shape[1]
        width = env.observation_space.shape[2]

        assert channels == 1 or channels == 3 or channels == 4
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(
            height, width, channels), dtype=env.observation_space.dtype)

    def __call__(self, state):
        return state.transpose((1, 2, 0))


'''
class observation_mode(Enum):
    gray = __OM_Gray
    interval = __OM_Interval
    crop = __OM_Crop
    resize = __OM_Resize
    chw = __OM_CHW
    default = __OM_Default
'''

mode = namedtuple('observation_transform', 'gray interval crop resize binary chw stack default')(
    __OM_Gray, __OM_Interval, __OM_Crop, __OM_Resize, __OM_Binary, __OM_CHW, __OM_Stack, __OM_Default)
