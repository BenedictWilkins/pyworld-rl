#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 14-06-2020 14:22:17

    File IO for various file formats.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"


from pyworld.toolkit.tools.fileutils.__import__ import fileio

class PickleIO(fileio):

    def __init__(self):
        super(PickleIO, self).__init__('.pickle', 'pickle')
    
    def save(self, file, data):
        with open(file, 'wb') as fp:
            self.pickle.dump(data, fp)

    def load(self, file):
        with open(file, 'rb') as fp:
            data = self.pickle.load(fp)
        return data

class TorchIO(fileio):

    class TorchIOException(Exception):
        pass

    def __init__(self):
        super(TorchIO, self).__init__('.pt', 'torch')
    
    def save(self, file, data):
        self.torch.save(data.state_dict(), file)

    def load(self, file, model=None):
        if model is None:
            raise TorchIO.TorchIOException("Loading a PyTorch model requires a template object, i.e. load(file, model=template)")
        model.load_state_dict(self.torch.load(file))
        return model #for some reason returning the model here can cause issues (it is not loaded fully before the return?) TODO ??
   
class HDF5IO(fileio):

    def __init__(self):
        super(HDF5IO, self).__init__('.hdf5', 'h5py')

    def save(self, file, data, chunk=None, groups=[], attrs={}, compress=True):
        #print("SAVE: ", file)
        if chunk is not None:
            raise NotImplementedError("TODO chunking is not implemented yet!")
        if len(attrs) > 0:
            raise NotImplementedError("TODO attrs not implemented yet!")
        
        f = self.h5py.File(file, "w")

        compression = [None, 'gzip'][int(compress)]

        for group in groups:
            group = f.create_group(group)

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = {"dataset":data}

        for k, d in data.items():
            dataset = f.create_dataset(str(k), data = d, compression=compression)

    def load(self, file):
        return self.h5py.File(file, 'r')



# TODO... do something?
def __load_mpz(path, max_size=100000):
    if os.path.isfile(path):
        data = np.load(path)
        for a in data:
            print(data[a])
            
        yield {k:v for k,v in np.load(path)}
    elif os.path.isdir(path): 
        fs = files(path)
        for f in fs:

            yield {k:v for k,v in np.load(f)}
    
def __save_mpz(path, data, z=False):
    if not z:
        np.savez(path, data)
    else:
        np.savez_compressed(path, data)
