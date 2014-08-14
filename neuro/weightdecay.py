import numpy
from reikna.algorithms.pureparallel import PureParallel
from reikna.core.signature import Parameter, Annotation
import neuro.kernels as kernels

import logging
log = logging.getLogger("weightdecay")

def renormalize_kernel(ctx, array, norm, constraint):
    kernel_cache, thread = ctx.kernel_cache, ctx.thread
    
    constraint = numpy.float32(constraint)
    
    key = (renormalize_kernel, array.shape, norm.shape, thread._context)
    if key not in kernel_cache.keys():
        comp = PureParallel(
            [
                Parameter('array', Annotation(array, 'io')),
                Parameter('norm', Annotation(norm, 'i')),            
                Parameter('constraint', Annotation(constraint))
            ],
        """
        // Renormalize if necessary
        float n = ${norm.load_idx}(${idxs[1]});
        float c = ${constraint};
        if ( n > c ) {  
            float a = ${array.load_same};
            a = a * c / n;
            ${array.store_same}(a);
        }
            
        """, guiding_array='array')

        kernel_cache[key] = comp.compile(thread)
        
    kernel_cache[key](array, norm, constraint)

class WeightNoise(object):    
    """
    Adds normal noise to the weight vector after each step
    """
    def __init__(self, **kwargs):
        super(WeightNoise, self).__init__(**kwargs)
        log.info("Renormalize constructor")
        
        self.weight_noise = []
        for w,b in kwargs['network'].weights:
            wn = self.context.thread.empty_like(w)
            bn = self.context.thread.empty_like(b)
            self.weight_noise.append((wn, bn))
    
    def update_weights(self, state):
        super(WeightNoise, self).update_weights(state)
        ctx = self.context
        for i in range(len(self.network.weights)):
            w, _ = self.network.weights[i]
            (wn, _) = self.weight_noise[i]
            ctx.normal(wn, 0.0, 0.0001)
            ctx.add(w, wn, w)
    
class Renormalize(object):    
    """
    Weight decay that renormalizes the weight vector of each neuron to a maximum length.
    """
    def __init__(self, **kwargs):
        super(Renormalize, self).__init__(**kwargs)
        log.info("Renormalize constructor")
        self.max_weight_size = kwargs.get('max_weight_size', 15.0)
        self.weight_norms = []
        for w,_ in kwargs['network'].weights:
            wn = self.context.thread.array(w.shape[-1:], dtype=w.dtype)
            self.weight_norms.append(wn)
    
    def update_weights(self, state):
        super(Renormalize, self).update_weights(state)
        ctx = self.context
        for i in range(len(self.network.weights)):
            w,b = self.network.weights[i]
            wn = self.weight_norms[i]
            ctx.norm(w, wn, 2.0, axes=(0,))            
            renormalize_kernel(ctx, w, wn, self.max_weight_size)