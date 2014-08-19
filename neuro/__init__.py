import logging
from neuro.kernels import KernelContext

import numpy

import reikna.cluda as cluda


log = logging.getLogger("neuro")

def create(name, base, *mixins):
    '''
    Factory method. Optionally provide an arbitrary number of mixin classes.
    :param thread: the mainthread (from reikna)
    :param name: the name of the new class
    :param base: the base class
    '''
    def _mixinFactory(name, mixin, base):
        class _tmp(base, mixin):
            pass
        _tmp.__name__ = name
        return _tmp
    
    cls = base
    for mixin in mixins:
        cls = _mixinFactory(name, cls, mixin)
    
    return cls

class BaseContext(KernelContext):
    
    def __init__(self):
        super(BaseContext, self).__init__()
        
        self.thread = self.api.Thread.create()
        self.kernel_cache = {}

            
    def synchronize(self):
        self.thread.synchronize()
    
    def upload(self, array, *arrays):
        thread = self.thread
        if isinstance(arrays, tuple) and len(arrays) > 0:          
            arrays = (array,) + arrays
            return (thread.to_device(arr) for arr in arrays)
        else:          
            return thread.to_device(array)

shape_cache = {}
def reshape(array, shape):
    '''
    Reshape function. Caches array objects so one shape only gets created once.
    :param array:
    :param shape:
    '''
    
    if array.shape != shape:
        key = (array.gpudata, shape)
        if not shape_cache.has_key(key):
            shape_cache[key] = array.reshape(shape)
        return shape_cache[key]
    else:
        return array