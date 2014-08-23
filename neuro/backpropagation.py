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

        # this can be different for every network implementation!
        net.delta(state, targets)
        
        # start at the output layer, end at the input layer
        for i in range(1, len(state.activations)):
            # get the current layer   
            layer = net.layers[-i]

            # apply the derivative to the activations
            # attention: activations contain f(x), not x!
            # so the derivative implementation must consider this.
            layer.derivative(state, -i)
            
            # calculate the gradient
            layer.calculate_gradient(state, -i)
     
            # backpropagate deltas to the previous layer
            # this is not necessary for the input layer
            if i < len(state.activations)-1:
                # backpropagate the error through the weights        
                layer.backpropagate(state, -i)
