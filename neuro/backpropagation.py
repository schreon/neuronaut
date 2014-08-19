import logging
log = logging.getLogger("backpropagation")

class Backpropagation(object):
    '''
    Backpropagation mixin. Defines how the gradient is to be calculated.
    '''
    def __init__(self, **kwargs):
        super(Backpropagation, self).__init__(**kwargs)
        log.info("Backpropagation constructor")
    
    def calculate_gradient(self, inputs, targets, state):        
        '''
        Calculate the gradient using the Backpropagation algorithm.
        
        :param inputs: input patterns
        :param targets: target values
        :param state: network state
        '''
        net = self.network
                
        deltas = state.deltas
        activations = state.activations
        gradients = state.gradients
        
        # this can be different for every network implementation!
        net.delta(state, targets)
        
        # start at the output layer, end at the input layer
        for i in range(1, len(activations)):
            # get the current layer   
            layer = net.layers[-i]                     
            # get the activations of this layer
            this_activations = activations[-i]
            # get the activations of the previous layer
            prev_activations = activations[-i-1]
            # get the deltas of this layer
            delta = deltas[-i]
            # get the correct weight arrays
            gradient_weights, gradient_bias = gradients[-i]
            
            # apply the derivative to the activations
            # attention: activations contain f(x), not x!
            # so the derivative implementation must consider this.
            layer.derivative(this_activations, delta)
            
            # calculate the gradient
            layer.calculate_gradient(prev_activations, delta, gradient_weights, gradient_bias)
     
            # backpropagate deltas to the previous layer
            # this is not necessary for the input layer
            if i < len(activations)-1:    
                # get the deltas of the previous layer      
                prev_delta = deltas[-i-1]
                # get the weights of this layer                
                weights, _ = net.weights[-i]
                
                # backpropagate the error through the weights        
                layer.backpropagate(delta, weights, prev_delta)                
