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



def convert_weights(net):
    pass


class BaseContext(KernelContext):
    
    def __init__(self, num_workers=10):
        super(BaseContext, self).__init__()
        
        self.thread = self.api.Thread.create()
        self.kernel_cache = {}
        
        self.workers = []
        for _ in xrange(num_workers):
            self.workers.append(self.api.Thread(self.thread._context))
        
        self.current_worker = 0
    
    def get_worker(self):
        self.current_worker = (self.current_worker + 1) % len(self.workers)
        return self.workers[self.current_worker]
    
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
        
class OpenClContext(BaseContext):
    '''
    OpenCL context.
    '''
    def __init__(self, num_workers=10): 
        self.api = cluda.ocl_api()  
        super(OpenClContext, self).__init__(num_workers=num_workers)

