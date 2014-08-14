import logging
import numpy

log = logging.getLogger("model")
    
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
        self.functions = []
        self.derivatives = []
        
    def add_layer(self, output_shape, transfer_function, transfer_derivative, **kwargs):
        """
        Add a layer to the neural network.
        This creates weight and bias matrices.
        """
        if not isinstance(output_shape, tuple):
            output_shape = (output_shape,)
        input_shape = self.shape[-1]        
        self.shape += (output_shape,)
        weight_shape = input_shape + output_shape      
        thr = self.context.thread        
        weights = thr.array(weight_shape, dtype=numpy.float32)
        bias = thr.array(output_shape, dtype=numpy.float32)        
        self.weights.append((weights, bias))
        self.functions.append(transfer_function)
        self.derivatives.append(transfer_derivative)

    def propagate(self, state, inputs, **kwargs):
        '''
        Propagates the given inputs through the network.
        :param state: The state object where intermediate results are to be stored.
        :param inputs: The input patterns.
        '''
        
        ctx = self.context
        # replace the first element with the new inputs
        state.activations[0] = inputs
        activations = state.activations 
        functions = self.functions
        
        for i in range(len(activations)-1):
            self.before_propagation(i, state, **kwargs)
            
        for i in range(len(activations)-1):            
            weights, bias = self.weights[i]            
            ctx.dot(activations[i], weights, activations[i+1])
            
            self.before_activation(i+1, state, **kwargs)
            functions[i](activations[i+1], bias)
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

    def mse(self, inputs, targets, state):   
        """
        Calculate the mean squared error on the given inputs/targets pairs
        """
        self.propagate(state, inputs)
        self.delta(state, targets)  
        self.context.norm(state.deltas[-1], state.error, 2.0)                         
        return state.error.get()[0]**2 / state.size

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
        for shape in net.shape[1:]:
            activation_shape = (self.size,) + shape
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
