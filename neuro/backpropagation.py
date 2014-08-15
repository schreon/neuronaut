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
        ctx = self.context
        net = self.network
        
        derivatives = net.derivatives
        
        deltas = state.deltas
        activations = state.activations
        gradients = state.gradients
        
        # this can be different for every network implementation!
        net.delta(state, targets)
        
        # start at the output layer, end at the input layer
        for i in range(1, len(activations)):
            # get the derivative function of this layer
            derivative = derivatives[-i]
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
            derivative(this_activations, delta)
            
            # calculate the gradients        
            ctx.dot(prev_activations, delta, gradient_weights, trans_a=True)
            
            # bias gradient is just the sum of the deltas
            # (because bias activation is implicitly 1.0)
            ctx.sum(delta, gradient_bias, axis=0)
                        
            # backpropagate deltas to the previous layer
            # this is not necessary for the input layer
            if i < len(activations)-1:    
                # get the deltas of the previous layer      
                prev_delta = deltas[-i-1]
                # get the weights of this layer                
                weights, _ = net.weights[-i]
                # backpropagate the error through the weights                
                ctx.dot(delta, weights, prev_delta, trans_b=True)
