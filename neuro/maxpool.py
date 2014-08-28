import logging
from neuro import reshape
from neuro.model import LayerState

import numpy

log = logging.getLogger("maxpool")

class Maxpool2DState(LayerState):
    pass

class Maxpool2DLayer(object):
    
    def __init__(self, context, input_shape, filter_shape=(2,2), **kwargs):
        log.info("Maxpool2DLayer constructor")
        self.context = context
        thr = context.thread
        
        channels, d1, d2 = input_shape
        f1, f2 = filter_shape
        d1 = int(numpy.ceil(d1 / float(f1)))
        d2 = int(numpy.ceil(d2 / float(f2)))
        
        self.output_shape = (channels, d1, d2)

    def create_state(self, num_patterns, state=None):
        log.info("MaxPool create_state")
        thread = self.context.thread
        activation_shape = (num_patterns,) + self.output_shape
        log.info(activation_shape)
        if state is None:
            state = Maxpool2DState(activation_shape)
        state.activations = thread.array(activation_shape, numpy.float32)
        return state

    def initialize_training_state(self, training_state, **kwargs):
        return training_state

    def initialize_test_state(self, test_state):
        return test_state

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
