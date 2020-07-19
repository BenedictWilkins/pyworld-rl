#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 14-06-2020 14:02:07

    File IO for common video formats.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import cv2
from pyworld.toolkit.tools.fileutils.__import__ import fileio

class gifIO(fileio): #TODO use opencv or pill (avoid movie py dependancy)

    def __init__(self):
        super(gifIO, self).__init__('.gif', 'moviepy.editor')

    def save(self, file, video, fps=24, duration=None):
        sequence = [a for a in video]
        length = len(sequence) #iterator requires a length (must be finite)
        assert length > 0

        if duration is not None:
            durations = [duration/length for _ in range(length)]
            clip = self.moviepy.ImageSequenceClip(sequence, durations=durations)
        else:
            clip = self.moviepy.ImageSequenceClip(sequence, fps=fps)

        clip.write_gif(file, fps=fps, program='ffmpeg')

    def load(self, file):
        raise NotImplementedError("TODO!")

class mp4IO(fileio):

    def __init__(self):
        super(mp4IO, self).__init__('.mp4', 'cv2')

    def save(self, file, video, fps=24):
        #TODO ensure NHWC format
        if issubclass(video.dtype.type, np.integer):
            if video.dtype.type != np.uint8:
                video = video.astype(np.uint8)
        elif issubclass(video.dtype.type, np.floating):
            raise ValueError("Video must be in integer [0-255] format")
        
        #video must be CV format (NHWC)
        colour = len(video.shape) == 4 and video.shape[-1] == 3
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') #ehhh.... platform specific?
        
        writer = cv2.VideoWriter(file, fourcc, fps, (video.shape[2], video.shape[1]), colour)
        for frame in video:
            writer.write(frame)
    
    def load(self, file):
        raise NotImplementedError("TODO!")

#TODO

def __save_avi(file):
    raise NotImplementedError("TODO!")

def __load_avi(file, as_numpy=True): #will be read as NHWC int format
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    images = []
    while success:
        images.append(image)
        success, image = vidcap.read()
    return np.array(images)