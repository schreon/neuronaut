import logging
from neuro import reshape

import numpy

log = logging.getLogger("maxpool")

class Maxpool2DLayer(object):
    
    def __init__(self, context, input_shape, num_units=16, filter_shape=(2,2), **kwargs):
        log.info("Maxpool2DLayer constructor")
        self.context = context
        thr = context.thread
        
        d1, d2 = input_shape
        f1, f2 = filter_shape
        d1 = int(numpy.ceil(d1 / float(f1)))
        d2 = int(numpy.ceil(d2 / float(f2)))
        
        self.output_shape = (d1, d2)
        
        weights_shape = (num_units,) + filter_shape
        
        weights = thr.array(weights_shape, dtype=numpy.float32)
        bias = thr.array((num_units,), dtype=numpy.float32)       
         
        self.weights = weights
        self.bias = bias

    def propagate(self, activations, next_activations):
        pass

    def transfer(self, activations):
        pass

    def derivative(self, activations, delta):
        pass
    
    def calculate_gradient(self, prev_activations, delta, gradient_weights, gradient_bias): 
        pass
    
    def backpropagate(self, delta, weights, prev_delta):
        pass
