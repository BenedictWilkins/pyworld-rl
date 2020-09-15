#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:36:14 2019

@author: ben
"""

# W&B Imports

import os
import re
import wandb
import torch

from . import fileutils as fu

def dryrun(value=True):
    os.environ["WANDB_MODE"] = ("run", "dryrun")[int(value)]

def runs(project):
    try:
        api = wandb.Api()
        return [run for run in api.runs(project)]
    except:
        raise ValueError("Something went wrong retrieving the runs: ensure project is specified properly as \"username/projectname\".")

def dryruns(path='./wandb/', full=False):
    return sorted(fu.filter(fu.dirs(path, full=full), blacklist=[]))

def download_files(run, path=".", replace=False, skip=[]):
    print(run)
    def should_skip(file):
        return any(re.match(file, pattern) for pattern in skip)

    if isinstance(run, str):
        api = wandb.Api()
        run = api.run(run)

    files = run.files()
    print("-- DOWNLOADING {0} files from run {1} to folder {2}".format(len(files), run.name, path))
    f_len = str(len(str(len(files))))
    for i, file in enumerate(files):
        if not should_skip(file):
            print(("---- {0:<" + f_len  + "}/{1:<" + f_len + "} downloading {2}").format(i, len(files), file.name))
            try:
                file.download(root=path, replace=replace)
            except:
                print("---- file found locally") #??
        else:
            print("---- skipping file {0}".format(file))
    print("-- FINISHED")

class ModelFileTorch:

    def __init__(self, path):
        self.path = path

    def load(self, template):
        fu.load(self.path, model=template)
        return template

def load(run, path="./wandb/", replace=False, blacklist=[], whitelist=[], detailed=False):
    WANDB_META = "wandb-metadata.json"
    WANDB_MODE = "mode"
    WANDB_RUN = "run"
    WANDB_DRYRUN = "dryrun"
    WANDB_CONFIG = "config.yaml"
    WANDB_MODEL_EXTS = ["*.pt"] #torch... TODO update to contain others

    path = os.path.join(path, run)
    path = os.path.abspath(path)

    
    if os.path.isdir(path):
        print(" -- found local run at {0}".format(path))
        try:
            meta_data = fu.load(os.path.join(path, WANDB_META))
            if meta_data[WANDB_MODE] == WANDB_RUN and replace:
                download_files(run, path=path, replace=replace)
        except:
            print(" -- failed to find wandb meta data...")
            if replace:
                download_files(run, path=path, replace=replace)
    else:
        print(" -- failed to find file locally...")
        download_files(run, path=path, replace=replace)

    from pprint import pprint
    #load files
    config = fu.load(os.path.join(path, WANDB_CONFIG))
    print(" -- found config file.")
    #pprint(config)
    def value_or(v):
        try:
            return v['value']
        except:
            return v
    config = {k:value_or(v) for k,v in config.items()}
    if not detailed:
        del config['wandb_version']
        del config['_wandb']
        #any others?

    #pprint(config)
    models = fu.filter(fu.files(path, full=True), whitelist=WANDB_MODEL_EXTS)
    if len(models) == 0:
        raise FileNotFoundError("Failed to find any model files for run {0}".format(run))
    print(" -- found {0} model(s): ".format(len(models)))
    for model in models:
        print(" ---- {0}".format(os.path.split(model)[1]))
    models = {os.path.split(m)[1]:ModelFileTorch(m) for m in models}
    return models, config


class WB:
   
    def __init__(self, project, model, save=True, id=None, config={}, **options):
        self.project = project
        self.model = model
        self.__save = save
        if id is None:
            id = fu.file_datetime()
        self.__step = 0

        if not isinstance(config, dict):
            try:
                config = vars(config) #e.g. SimpleNamespace
            except:
                raise ValueError("Invalid argument: \"config\", should be of type dict.")

        config['model_class'] = self.model.__class__.__name__

        self.run = wandb.init(project=project, id=id, config=config, **options)

    def __enter__(self):
        print("wandb project: {0}".format(self.project))
        print("-- run: {0}".format(self.run))
        print("-- directory: {0}".format(os.path.split(self.run.dir)[1]))
        wandb.watch(self.model, log='all')
        
    def __call__(self, **info):
        wandb.log(info, step=self.__step)
    
    def step(self):
        self.__step += 1
        return self.__step

    def histogram(self, array):
        return wandb.Histogram(array)

    def image(self, array, name='image'):
        return [wandb.Image(array, caption=name)]

    def save(self, overwrite=True):
        print("--- saving model: step {0}".format(self.__step))
        file = os.path.join(wandb.run.dir, 'model.pt')
        if not overwrite:
            file = fu.file(file)
        torch.save(self.model.state_dict(), file)
        wandb.save(file)
        print("--- done.")
   
    def __exit__(self, type, value, traceback):
        if self.__save:
            self.save()
            
            
        


            

