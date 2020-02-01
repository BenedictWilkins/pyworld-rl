#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:36:14 2019

@author: ben
"""

# W&B Imports

import os
'''
#doesnt work... why not :(
if 'WB_ONLINE' in globals():
    __WANDB_MODE = {'offline':'dryrun', 'online':'run'}
    mode = globals()['WB_ONLINE']
    assert isinstance(mode, bool) #WANDB_ONLINE must be a boolean
    if mode:
        os.environ['WANDB_MODE'] = __WANDB_MODE['online']
    else:
        os.environ['WANDB_MODE'] = __WANDB_MODE['offline']
'''

import wandb
import torch

from . import fileutils as fu

def get_runs(project):
    try:
        api = wandb.Api()
        return [run for run in api.runs(project)]
    except:
        raise ValueError("Something went wrong retrieving the runs: ensure project is specified properly as \"username/projectname\".")

def download_files(run, path=".", replace=False):
    if isinstance(run, str):
        api = wandb.Api()
        run = api.run(run)

    files = run.files()
    print("-- DOWNLOADING {0} files from run {1} to folder {2}".format(len(files), run.name, path))
    f_len = str(len(str(len(files))))
    for i, file in enumerate(files):
        print(("---- {0:<" + f_len  + "}/{1:<" + f_len + "} downloading {2}").format(i, len(files), file.name))
        try:
            file.download(root=path, replace=replace)
        except:
            print("---- file found locally")
    print("-- FINISHED")

class WB:
   
    def __init__(self, project, model, save=True, id=None,  config={}, **options):
        self.project = project
        self.model = model
        self.__save = save
        if id is None:
            id = fu.file_datetime()
        self.__step = 0

        wandb.init(project=project, id=id, config = config, **options)

    def __enter__(self):
        wandb.watch(self.model, log='all')
        
    def __call__(self, **info):
        wandb.log(info, step=self.__step)
    
    def step(self):
        self.__step += 1
        return self.__step

    def image(self, array, name='image'):
        return [wandb.Image(array, caption=name)]

    def save(self, overwrite=True):
        file = os.path.join(wandb.run.dir, 'model.pt')
        print("wbsave: ", file)
        if not overwrite:
            file = fu.file(file)
        torch.save(self.model.state_dict(), file)
        wandb.save(file)
   
    def __exit__(self, type, value, traceback):
        if self.__save:
            self.save()
            
            
        


            

