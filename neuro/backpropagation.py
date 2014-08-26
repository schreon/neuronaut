import logging
log = logging.getLogger("backpropagation")

class BackpropagationTrainer(object):
    '''
    Backpropagation mixin. Defines how the gradient is to be calculated.
    '''
    def __init__(self, *args, **kwargs):
        super(BackpropagationTrainer, self).__init__(*args, **kwargs)
        log.info("Backpropagation constructor")
    
    def calculate_gradient(self, state, inputs, targets):
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
        for i in reversed(range(0, len(state.layers))):
            # get the current layer   
            layer = net.layers[i]
            layer_state = state.layers[i]
            if (i > 0):
                layer_inputs = state.layers[i-1].activations
            else:
                layer_inputs = inputs

            # apply the derivative to the activations
            # attention: activations contain f(x), not x!
            # so the derivative implementation must consider this.
            layer.derivative(layer_state)
            
            # calculate the gradient
            layer.calculate_gradient(layer_state, layer_inputs)
     
            # backpropagate deltas to the previous layer
            # this is not necessary for the input layer
            if i > 0:
                # backpropagate the error through the weights        
                layer.backpropagate(layer_state, state.layers[i-1].deltas)
