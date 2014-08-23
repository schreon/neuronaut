import logging
from neuro import reshape

import numpy


log = logging.getLogger("model")

class LogisticLayer(object):
    def __init__(self, context, *args, **kwargs):
        log.info("LogisticLayer constructor")
        super(LogisticLayer, self).__init__(context, *args, **kwargs)
        
        self.transfer_function = context.logistic
        self.transfer_derivative = context.logistic_derivative

class LinearLayer(object):
    def __init__(self, context, *args, **kwargs):
        log.info("LinearLayer constructor")
        super(LinearLayer, self).__init__(context, *args, **kwargs)
        
        self.transfer_function = context.linear
        self.transfer_derivative = context.linear_derivative

class DenseLayer(object):
    
    def __init__(self, context, input_shape, num_units=128, **kwargs):
        log.info("DenseLayer constructor")
        self.context = context
        thr = context.thread
        
        
        if not isinstance(input_shape, tuple):
            input_shape = (input_shape,)
        if len(input_shape) > 1:
            d = 1
            for dim in input_shape:
                d *= dim
            input_shape = (d,)
        
        self.input_shape = input_shape
        self.output_shape = (num_units,)
        
        weights_shape = input_shape + self.output_shape 
        
        weights = thr.array(weights_shape, dtype=numpy.float32)
        bias = thr.array((num_units,), dtype=numpy.float32)       
         
        self.weights = weights
        self.bias = bias

        self.targets_shape = self.output_shape

    def propagate(self, state, idx):
        activations = state.activations[idx]
        next_activations = state.activations[idx+1]

        desired_shape = activations.shape[:1] + self.input_shape
        activations = reshape(activations, desired_shape)
        self.context.dot(activations, self.weights, next_activations)
    
    def transfer(self, state, idx):
        activations = state.activations[idx]
        desired_shape = activations.shape[:1] + self.output_shape
        activations = reshape(activations, desired_shape)        
        self.transfer_function(activations, self.bias)

    def derivative(self, state, idx):
        activations = state.activations[idx]
        delta = state.deltas[idx]

        desired_shape = activations.shape[:1] + self.output_shape
        activations = reshape(activations, desired_shape)
        delta = reshape(delta, desired_shape)
        self.transfer_derivative(activations, delta)
    
    def calculate_gradient(self, state, idx):
        prev_activations = state.activations[idx-1]
        delta = state.deltas[idx]
        gradient_weights, gradient_bias = state.gradients[idx]

        desired_shape = prev_activations.shape[:1] + self.input_shape
        prev_activations = reshape(prev_activations, desired_shape)
          
        ctx = self.context
        # calculate the gradients        
        ctx.dot(prev_activations, delta, gradient_weights, trans_a=True)         
        # bias gradient is just the sum of the deltas
        # (because bias activation is implicitly 1.0)
        ctx.sum(delta, gradient_bias, axis=0)
    
    def backpropagate(self, state, idx):
        delta = state.deltas[idx]
        # get the deltas of the previous layer
        prev_delta = state.deltas[idx-1]

        weights = self.weights

        desired_shape = prev_delta.shape[:1] + self.input_shape
        prev_delta = reshape(prev_delta, desired_shape)
        ctx = self.context
        ctx.dot(delta, weights, prev_delta, trans_b=True)

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
        self.targets_dtype= numpy.float32
        self.error_measure = "RMSE"

    def add_layer(self, **kwargs):
        """
        Add a layer to the neural network.
        """
        ctx = self.context
        log.info(self.shape)
        input_shape = self.shape[-1] 
        LayerClass = kwargs['LayerClass']          
        new_layer = LayerClass(ctx, input_shape, **kwargs)        
        self.layers.append(new_layer)
             
        self.shape += (new_layer.output_shape,)

        # save additional references to the layers' weights
        self.weights.append((new_layer.weights, new_layer.bias))


    def propagate(self, state, inputs, **kwargs):
        '''
        Propagates the given inputs through the network.
        :param state: The state object where intermediate results are to be stored.
        :param inputs: The input patterns.
        '''
        
        # replace the first element with the new inputs
        state.activations[0] = inputs
        activations = state.activations 
        
        for i in range(len(activations)-1):
            self.before_propagation(i, state, **kwargs)
            
        for i in range(len(activations)-1):
            layer = self.layers[i]      
                    
            layer.propagate(state, i)
            
            self.before_activation(i+1, state, **kwargs)
            layer.transfer(state, i+1)
            self.after_activation(i+1, state, **kwargs)
            
        for i in range(len(activations)-1):
            self.after_propagation(i, state, **kwargs)
            
    def delta(self, state, targets):
        '''
        Calculate the error between the given target values and 
        the values calculated by the network.
        :param state: The state of the network.
        :param targets: The desired target values.
        '''

    def error(self, inputs, targets, state):
        """
        Calculate the mean squared error on the given inputs/targets pairs
        """
        self.propagate(state, inputs)
        self.delta(state, targets)  
        self.context.norm(state.deltas[-1], state.error, 2.0)
        return numpy.sqrt(state.error.get()[0]**2 / state.size)

    def reset_weights(self, std=0.01):
        '''
        Fill the weight matrices with random values from a normal distribution with the mean=0.0.
        The bias weights will be set to zero.
        
        :param std: the standard deviation of the normal distribution.
        '''
        for weights, bias in self.weights:
            self.context.normal(weights, 0.0, std, seed=self.seed)
            bias.fill(numpy.float32(0.0))
            
    def download_weights(self):
        wgts = []
        for weights, bias in self.weights:
            wgts.append((weights.get(), bias.get()))        
        return wgts
    
    def upload_weights(self, weights):
        for i, (w, b) in enumerate(self.weights):
            (w_cpu, b_cpu) = weights[i]
            w.set(w_cpu)
            b.set(b_cpu)     
        
    def before_propagation(self, layer_index, state, **kwargs):
        pass
    
    def after_propagation(self, layer_index, state, **kwargs):
        pass
    
    def before_activation(self, layer_index, state, **kwargs):
        pass
    
    def after_activation(self, layer_index, state, **kwargs):
        pass

class NetworkState(object):
    '''
    Holds the state belonging to a neural network. 
    '''
    
    def __init__(self, **kwargs):        
        super(NetworkState, self).__init__()
        log.info("NetworkState constructor")        
        net = kwargs['network']
        self.size = kwargs['size']

        thread = net.context.thread
        
        # create activation arrays.
        # the first activation array will
        # be given as the inputs array.
        # the first element is reserved for the inputs.  
        self.activations = [None]      
        for layer in net.layers:
            activation_shape = (self.size,) + layer.output_shape
            act = thread.array(activation_shape, numpy.float32)
            self.activations.append(act)
        
        
class NaNMask(object):
    """
    Replaces NaN values in input with zeros.
    NaN target values are already handles in the sub kernel.
    There, a difference containing a nan value will always result in zero.
    """

    def __init__(self, **kwargs):        
        super(NaNMask, self).__init__(**kwargs)
        log.info("NaNMask constructor")
       
    def propagate(self, state, inputs, **kwargs):
        """
        Before the inputs are propagated, all nan values are replaced by zeros.
        """
        self.context.nan_to_zeros(inputs, inputs)
        super(NaNMask, self).propagate(state, inputs, **kwargs)    

class Regression(object):
    """
    Defines the ouput of a neural network to solve a regression task.
    """

    def __init__(self, **kwargs):        
        super(Regression, self).__init__(**kwargs)
        log.info("Regression constructor")

    
    def delta(self, state, targets):
        """
        The error is the difference (targets - netoutput).
        """        
        super(Regression, self).delta(state, targets)
        self.context.sub(targets, state.activations[-1], state.deltas[-1])
