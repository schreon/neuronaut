import logging
from neuro import reshape

import numpy

log = logging.getLogger("classification")


def classification_delta_kernel(ctx, inputs, targets, deltas):
    pass

class Classification(object):
    """
    Defines the ouput of a neural network to solve a regression task.
    """

    def __init__(self, **kwargs):        
        super(Classification, self).__init__(**kwargs)
        log.info("Regression constructor")

    
    def delta(self, state, targets):
        """
        Classes must be coded as integers. Each integer is one class.
        """        
        super(Classification, self).delta(state, targets)
        classification_delta_kernel(self.context, targets, state.activations[-1], state.deltas[-1])
