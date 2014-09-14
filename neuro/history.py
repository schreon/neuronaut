from neuro.training import Trainer

from neuro import kernels
import logging
import neuro
import numpy
import time
log = logging.getLogger("history")

class History(object):
    ''' Logs the history of a training session '''

    def __init__(self, *args, **kwargs):
        super(History, self).__init__(*args, **kwargs)
        log.info("History constructor")

        self.errors = {
           'current' : {
                        'test' : numpy.inf,
                        'train' : numpy.inf
                        },
           'best' : {
                        'test' : numpy.inf,
                        'train' : numpy.inf
                        },
           'history' : {
                        'test' : [],
                        'train' : []
                        }
           }

        self.validation_frequency = kwargs.get('validation_frequency', 5)
        self.validate_train = kwargs.get('validate_train', False) # MSE also on the training set?
        self.logging_frequency = kwargs.get('logging_frequency', 1.0)
        self.out_norm = kwargs.get('out_norm', (0.0, 1.0))
        self.start_time = None
        self.next_update = 0

    def train_step(self, *args, **kwargs):
        super(History, self).train_step(*args, **kwargs)

        network = self.network

        if self.test_data is not None and self.steps % self.validation_frequency == 0:
            inputs_test, targets_test = self.get_test_patterns()
            error_test = network.error(inputs_test, targets_test, self.test_state)
            self.errors['current']['test'] =  error_test
            self.errors['history']['test'].append(self.errors['current']['test'])

            if error_test < self.errors['best']['test']:
                self.best_step = self.steps
                old_best = self.errors['best']['test']
                self.errors['best']['test'] = error_test
                self.on_new_best(old_best, error_test)

            if self.validate_train:
                inputs, targets = self.get_training_patterns()
                error_train = network.error(inputs, targets, self.training_state)
                self.errors['current']['train'] =  error_train
                self.errors['history']['train'].append(self.errors['current']['train'])

        # some warmup steps are necessary because the
        # network first needs to be compiled etc.
        if self.steps < 5:
            self.start_time = time.time()
            self.next_update = self.start_time + self.logging_frequency
        else:
            # Measure performance
            current_time = time.time()
            self.steps_per_sec = (self.steps-5) / (current_time - self.start_time)
            if current_time > self.next_update:
                log.info("(%d)%s: best %.4f, current %.4f, train %.4f, %.2f steps / sec" % (self.steps, network.error_measure, self.errors['best']['test'], self.errors['current']['test'], self.errors['current']['train'], self.steps_per_sec))
                self.next_update += self.logging_frequency
