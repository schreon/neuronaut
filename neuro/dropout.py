import logging

import numpy
from reikna.algorithms.pureparallel import PureParallel
from reikna.core.signature import Parameter, Annotation

import neuro


log = logging.getLogger("dropout")

DROPOUT_TEST = 0
DROPOUT_TRAIN = 1

def dropout(ctx, mat, rand, probability):
    kernel_cache = ctx.kernel_cache
    probability = numpy.float32(probability)
    thread = ctx.thread
    key = (dropout, mat.dtype, mat.shape)

    if key not in kernel_cache.keys():
        log.info("compiling " + str(key))
        kernel = PureParallel(
            [
                Parameter('mat', Annotation(mat, 'o')),
                Parameter('rand', Annotation(mat, 'i')),
                Parameter('probability', Annotation(probability))
            ],
            """
        ${rand.ctype} r = ${rand.load_same};
        if (r < ${probability}) {            
            ${mat.store_same}(0.0f);
        }
        """, guiding_array='mat')

        kernel_cache[key] = kernel.compile(thread)

    kernel_cache[key](mat, rand, probability)

class DropoutState(object):
    '''
    State belonging to a dropout network.
    '''
    def __init__(self, **kwargs):        
        super(DropoutState, self).__init__(**kwargs)
        net = kwargs['network']
        ctx = net.context
        thread = ctx.thread
        self.dropout_rands = []  
        for shape in net.shape:
            rand_shape = (self.size,) + shape
            rand = thread.array(rand_shape, numpy.float32)     
            self.dropout_rands.append(rand)

class Dropout(object):
    '''
    Dropout Mixin for a Neural Network Trainer.
    '''
    def __init__(self, **kwargs):
        log.info("Dropout constructor")
        super(Dropout, self).__init__(**kwargs)
        self.TrainingState = neuro.create("TrainingState", self.TrainingState, DropoutState)
    
    def train_step(self, training_state, inputs, targets, **kwargs):
        kwargs['dropout_mode'] = DROPOUT_TRAIN
        super(Dropout, self).train_step(training_state, inputs, targets, **kwargs)

class DropoutNetwork(object):
    """
    Dropout Mixin for a Neural Network Model.
    
    Randomly turns units off with a given probability.
    """

    def __init__(self, **kwargs):        
        super(DropoutNetwork, self).__init__(**kwargs)
        log.info("Dropout constructor")
        input_dropout = kwargs.get('input_dropout', 0.0)
        self.dropout_probabilities = [input_dropout]
    
    def add_layer(self, *args, **kwargs):
        super(DropoutNetwork, self).add_layer(*args, **kwargs)
        dropout_probability = kwargs.get('dropout', 0.0)
        self.dropout_probabilities.append(dropout_probability)
        
    def before_propagation(self, layer_index, state, **kwargs):
        '''
        In Training mode, before the activations of a dropout-layer
        are propagated forward, the weights must be increased because
        only a proportion of (1.0 - dropout_probability) will be active.
        
        :param layer_index:
        :param state:
        '''
        super(DropoutNetwork, self).before_propagation(layer_index, state, **kwargs)
        probability = self.dropout_probabilities[layer_index]        
        mode = kwargs.get('dropout_mode', DROPOUT_TEST)
        if mode == DROPOUT_TRAIN:
            if probability > 0.0:
                weights, _ = self.weights[layer_index]           
                # scale weights up
                self.context.scale(weights, 1.0 / (1.0 - probability))    
    
    def after_activation(self, layer_index, state, **kwargs):
        '''
        After the activation/transfer function has been applied, the 
        neurons are dropped out with a certain probability.
        (See page 3 in http://www.cs.toronto.edu/~nitish/msc_thesis.pdf)
        
        :param layer_index:
        :param state:
        '''
        super(DropoutNetwork, self).after_activation(layer_index, state, **kwargs)
        probability = self.dropout_probabilities[layer_index]        
        mode = kwargs.get('dropout_mode', DROPOUT_TEST)
        if mode == DROPOUT_TRAIN:
            if probability > 0.0:
                activations = state.activations[layer_index]
                rand = state.dropout_rands[layer_index]                    
                # apply dropout                        
                self.context.uniform(rand, 0.0, 1.0)
                dropout(self.context, activations, rand, probability)

    def after_propagation(self, layer_index, state, **kwargs):
        '''
        In Training mode, after the propagation of the activations,
        the weights must be scaled down again.
        
        :param layer_index:
        :param state:
        '''
        super(DropoutNetwork, self).after_propagation(layer_index, state, **kwargs)
        probability = self.dropout_probabilities[layer_index]
        mode = kwargs.get('dropout_mode', DROPOUT_TEST)
        if mode == DROPOUT_TRAIN:         
            if probability > 0.0:
                weights, _ = self.weights[layer_index]
                # scale weights down
                self.context.scale(weights, (1.0 - probability))