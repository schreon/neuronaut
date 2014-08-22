



from neuro import kernels
import logging
import neuro
import numpy
import time
log = logging.getLogger("training")

class TestState(object):
    '''
    Holds the state belonging to a neural network during testing. 
    '''
    
    def __init__(self, **kwargs):        
        super(TestState, self).__init__(**kwargs)
        log.info("TestState constructor")    
        net = kwargs['network']
        thread = net.context.thread
        
        # only the deltas of the output layer are necessary
        # for testing purposes
        deltas_shape = (self.size,) + net.layers[-1].output_shape
        deltas = thread.array(deltas_shape, numpy.float32)
        self.deltas = [deltas]
        
        # in this array, the current error (for example MSE) will be stored
        self.error = thread.array((1,), dtype=net.targets_dtype)

class TrainingState(object):
    '''
    Holds the state belonging to a neural network during training. 
    '''
    
    def __init__(self, **kwargs):        
        super(TrainingState, self).__init__(**kwargs)
        log.info("TrainingState constructor")    
        net = kwargs['network']
        self.context = net.context
        thread = net.context.thread
        
        # arrays for the deltas are necessary for all layers
        # except the input layer        
        self.deltas = []
        for layer in net.layers:
            deltas_shape = (self.size,) + layer.output_shape
            deltas = thread.array(deltas_shape, numpy.float32)
            self.deltas.append(deltas)
        
        # for every weight array, a gradient array is necessary
        self.gradients = []
        for weights, bias in net.weights:
            gradients = thread.array(weights.shape, numpy.float32)
            gradients_bias = thread.array(bias.shape, numpy.float32)
            self.gradients.append((gradients, gradients_bias))
        
        # in this array, the current error (e.g. MSE) will be stored    
        self.error = thread.array((1,), dtype=net.targets_dtype)

class FullBatchTrainer(object):
    '''
    Full Batch Trainer.
    '''
    
    def __init__(self, **kwargs):
        log.info("FullBatchTrainer constructor")
        self.network = kwargs['network']
        self.seed = self.network.seed
        self.context = self.network.context
        self.steps = 0
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
        self.min_steps = kwargs.get('min_steps', 1000)
        self.validation_frequency = kwargs.get('validation_frequency', 5)
        self.validate_train = kwargs.get('validate_train', False) # MSE also on the training set?
        self.logging_frequency = kwargs.get('logging_frequency', 1.0)
        
        self.TrainingState = neuro.create("TrainingState", neuro.model.NetworkState, neuro.training.TrainingState)
        self.TestState = neuro.create("TestState", neuro.model.NetworkState, neuro.training.TestState)

    def is_finished(self):
        """ Stop as soon as the maximum number of steps has been reached """
        if self.steps >= self.min_steps:
            return True
        else:
            return False
    
    def train_step(self, training_state, inputs, targets, **kwargs):
        """ Perform a training step """
        self.network.propagate(training_state, inputs, **kwargs)
        self.network.delta(training_state, targets)                        
        self.calculate_gradient(inputs, targets, training_state)
        self.update_weights(training_state)
    

    def train(self, training_state, test_state, inputs, targets, inputs_test, targets_test, **kwargs):
        """ Train until the stopping criterion is met """
        network = self.network
        warm_up = 10 # number of steps until time is measured
        while not self.is_finished():
            # some warmup steps are necessary because the
            # network first needs to be compiled etc.
            if self.steps < warm_up:
                start_time = time.time()
                next_update = start_time + self.logging_frequency

            # Training Step
            self.train_step(training_state, inputs, targets, **kwargs)
            
            if test_state is not None and self.steps % self.validation_frequency == 0:
                error_test = network.error(inputs_test, targets_test, test_state)
                self.errors['current']['test'] =  error_test
                self.errors['history']['test'].append(self.errors['current']['test'])
                
                if self.validate_train:
                    error_train = network.error(inputs, targets, training_state)
                    self.errors['current']['train'] =  error_train
                    self.errors['history']['train'].append(self.errors['current']['train'])
            self.steps += 1
            
            # Measure performance
            current_time = time.time()
            self.steps_per_sec = (self.steps-warm_up) / (current_time - start_time)
            if current_time > next_update:                  
                log.info("(%d)%s: best %.4f, current %.4f, train %.4f, %.2f steps / sec" % (self.steps, network.error_measure, self.errors['best']['test'], self.errors['current']['test'], self.errors['current']['train'], self.steps_per_sec))
                next_update += self.logging_frequency


    def update_weights(self, state):
        """ Update the weight parameters of the network """
        pass        
    
    def calculate_gradient(self, inputs, targets, state):  
        """ Calculate the gradient for the network weights """
        pass

class SGDState(object):
    '''
    Holds the state belonging to a network trained by Stochastic Gradient Descent
    '''
    def __init__(self, *args, **kwargs):
        super(SGDState, self).__init__(*args, **kwargs)  
        
        thread = self.context.thread
        net = kwargs['network']
        
        self.indices = thread.array((self.size,), dtype=numpy.int32)
        self.inputs = thread.array((self.size,)+net.layers[0].input_shape, dtype=numpy.float32)
        self.targets = thread.array((self.size,)+net.layers[-1].targets_shape, dtype=net.targets_dtype)
        
class SGDTrainer(FullBatchTrainer):
    '''
    Stochastic Gradient Descent trainer
    '''
    
    def train_step(self, training_state, inputs, targets, **kwargs):
        # sample random minibatch
        self.context.randint(training_state.indices, 0, inputs.shape[0], seed=self.seed) 
        
        # copy minibatch      
        self.context.copy_minibatch(inputs, training_state.indices, training_state.inputs)
        self.context.copy_minibatch(targets, training_state.indices, training_state.targets)
        
        # pass to super train step
        return super(SGDTrainer, self).train_step(training_state, training_state.inputs, training_state.targets, **kwargs)

    def __init__(self, *args, **kwargs):        
        super(SGDTrainer, self).__init__(*args, **kwargs)        
        self.minibatch_size = kwargs.get('minibatch_size', 128)        
        self.TrainingState = neuro.create("TrainingState", self.TrainingState, SGDState)
