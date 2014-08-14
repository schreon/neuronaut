import logging

from neuro.kernels import KernelContext
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
        for w in self.workers:
            w.synchronize()
        self.thread.synchronize()
    
    def upload(self, array, *arrays):
        thread = self.thread
        if isinstance(arrays, tuple) and len(arrays) > 0:          
            arrays = (array,) + arrays
            return (thread.to_device(arr) for arr in arrays)
        else:          
            return thread.to_device(array)
