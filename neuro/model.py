import logging
from neuro import reshape

import numpy


log = logging.getLogger("model")

class LayerState(object):

    def __init__(self, shape):
        super(LayerState, self).__init__()
        self.shape = shape


class NetworkState(object):
    '''
    Holds the state belonging to a neural network.
    '''

    def __init__(self, **kwargs):
        super(NetworkState, self).__init__()
        log.info("NetworkState constructor")
        self.num_patterns = kwargs['num_patterns']

        self.layers = []

class FeedForwardNeuralNetwork(object):
    """
    A basic feed forward neural network.
    Use Mixins (e.g. Regression, NaNMask) to define further
    properties.
    """
    
    def __init__(self, **kwargs):        
        super(FeedForwardNeuralNetwork, self).__init__()
        log.info("FeedForwardNeuralNetwork constructor")
        self.seed = kwargs.get('seed', None)
        self.context = kwargs['context']
        input_shape = kwargs['input_shape']
        if not isinstance(input_shape, tuple):
            input_shape = (input_shape,)
        self.shape = (( input_shape,))      
        self.weights = []
        self.layers = []
        self.error_measure = "RMSE"

    def create_state(self, num_patterns):
        state = NetworkState(num_patterns=num_patterns)

        for layer in self.layers:
            state.layers.append(layer.create_state(num_patterns))

        # in this array, the current error (for example MSE) will be stored
        state.error = self.context.thread.array((1,), dtype=self.get_target_dtype())

        return state

    def add_layer(self, LayerClass, **kwargs):
        """
        Add a layer to the neural network.
        """
        ctx = self.context
        log.info(self.shape)
        input_shape = self.shape[-1]
        new_layer = LayerClass(ctx, input_shape, **kwargs)        
        self.layers.append(new_layer)
             
        self.shape += (new_layer.output_shape,)



    def propagate(self, state, inputs, **kwargs):
        '''
        Propagates the given inputs through the network.
        :param state: The state object where intermediate results are to be stored.
        :param inputs: The input patterns.
        '''

        #for layer, layer_state in zip(self.layers, state.layers):
        #    self.before_propagation(layer, layer_state, **kwargs)
            
        for layer, layer_state in zip(self.layers, state.layers):
            layer.propagate(layer_state, inputs)
            
            #self.before_activation(layer, layer_state, **kwargs)
            layer.transfer(layer_state)
            #self.after_activation(layer, layer_state, **kwargs)

            inputs = layer_state.activations
            
        #for layer, layer_state in zip(self.layers, state.layers):
        #    self.after_propagation(layer, layer_state, **kwargs)
            
    def delta(self, state, targets):
        '''
        Calculate the error between the given target values and 
        the values calculated by the network.
        :param state: The state of the network.
        :param targets: The desired target values.
        '''

    def reset(self, std=0.01):
        '''
        Fill the weight matrices with random values from a normal distribution with the mean=0.0.
        The bias weights will be set to zero.
        
        :param std: the standard deviation of the normal distribution.
        '''
        for layer in self.layers:
            layer.reset(std=0.01)
            
    def download(self):
        return [layer.download() for layer in self.layers]
    
    def upload(self, weights):
        for i, w in enumerate(self.weights):
            self.layers[i].upload(w)
        
    def before_propagation(self, layer, layer_state, **kwargs):
        pass
    
    def after_propagation(self, layer, layer_state, **kwargs):
        pass
    
    def before_activation(self, layer, layer_state, **kwargs):
        pass
    
    def after_activation(self, layer, layer_state, **kwargs):
        pass


class NaNMask(object):
    """
    Replaces NaN values in input with zeros.
    NaN target values are already handles in the sub kernel.
    There, a difference containing a nan value will always result in zero.
    """

    def __init__(self, **kwargs):        
        super(NaNMask, self).__init__(**kwargs)
        log.info("NaNMask constructor")
       
    def propagate(self, inputs, states, **kwargs):
        """
        Before the inputs are propagated, all nan values are replaced by zeros.
        """
        self.context.nan_to_zeros(inputs, inputs)
        super(NaNMask, self).propagate(inputs, states, **kwargs)

class RegressionNetwork(object):
    """
    Defines the ouput of a neural network to solve a regression task.
    """

    def __init__(self, **kwargs):        
        super(RegressionNetwork, self).__init__(**kwargs)
        log.info("Regression constructor")

    def create_state(self, num_patterns):
        state = super(RegressionNetwork, self).create_state(num_patterns)
        ctx = self.context
        shp = (num_patterns,) + self.shape[-1]
        state.regression_errors = ctx.thread.array(shp, dtype=numpy.float32)
        return state
    
    def delta(self, network_state, targets):
        """
        The error is the difference (targets - netoutput).
        """        
        super(RegressionNetwork, self).delta(network_state, targets)

        outputs = network_state.layers[-1].activations
        deltas = network_state.layers[-1].deltas

        self.context.sub(targets, outputs, deltas)

    def get_target_dtype(self):
        return numpy.float32

    def error(self, inputs, targets, network_state):
        """
        Calculate the mean squared error on the given inputs/targets pairs
        """
        self.propagate(network_state, inputs)
        outputs = network_state.layers[-1].activations
        error = network_state.error
        deltas = network_state.regression_errors

        self.context.sub(targets, outputs, deltas)
        self.context.norm(deltas, error, 2.0)
        return numpy.sqrt(error.get()[0]**2 / targets.shape[0]*targets.shape[1])