

import logging
log = logging.getLogger("stopping")
class EarlyStopping(object):
    '''
    Early Stopping. Makes the training process stop if there
    has not been found a new optimum on a validation set for
    the same amount of steps that have been necessary to 
    reach the last optimum.
    '''
    def __init__(self, **kwargs):
        super(EarlyStopping, self).__init__(**kwargs)
        log.info("EarlyStopping constructor")
        
    def is_finished(self):
        if self.errors['current']['test'] < self.errors['best']['test']:            
            self.min_steps = max(self.min_steps, 2 * self.steps + 1)
            self.errors['best']['test'] = self.errors['current']['test']
        if self.steps < self.min_steps:            
            return False
        else:          
            log.info("Early Stopping - finished after %d steps" % self.steps)              
            return True
