import logging

import numpy

log = logging.getLogger("training")

class Trainer(object):
    '''
    Holds the state of a Trainer Session.
    '''

    def __init__(self, network=None, training_data=None, test_data=None, **kwargs):
        log.info("Trainer constructor")
        if network is None:
            raise Exception("the 'network' keyword parameter must be specified")
        if network is None:
            raise Exception("the 'training_data' keyword parameter must be specified")

        self.network = network
        self.context = network.context
        self.training_data = training_data
        self.test_data = test_data

        self.steps = 0
        self.min_steps = kwargs.get('min_steps', 1000)

        self.test_state = network.create_state(self.get_num_test_patterns())
        self.training_state = network.create_state(self.get_batch_size())

        # Layer State initialization
        for layer, layer_state in zip(self.network.layers, self.training_state.layers):
            layer.initialize_training_state(layer_state)

        # Trainer State initialization
        self.initialize_test_state(self.test_state, **kwargs)
        self.initialize_training_state(self.training_state, **kwargs)

    def initialize_training_state(self, training_state, **kwargs):
        pass

    def initialize_test_state(self, test_state):
        pass

    def get_batch_size(self):
        return self.training_data[0].shape[0]

    def get_num_training_patterns(self):
        return self.training_data[0].shape[0]

    def get_num_test_patterns(self):
        return self.test_data[0].shape[0]

    def on_new_best(self, old_best, new_best):
        pass

    def is_finished(self):
        """ Stop as soon as the maximum number of steps has been reached """
        if self.steps >= self.min_steps:
            return True
        else:
            return False

    def get_training_patterns(self):
        return self.training_data

    def get_test_patterns(self):
        return self.test_data

    def train_step(self, **kwargs):
        """ Perform a training step """

        inputs, targets = self.get_training_patterns()

        self.network.propagate(self.training_state, inputs, **kwargs)
        self.network.delta(self.training_state, targets)
        self.calculate_gradient(self.training_state, inputs, targets)
        self.update_weights(self.training_state)

    def train(self, **kwargs):
        """ Train until the stopping criterion is met """
        network = self.network

        while not self.is_finished():
            # Training Step
            self.train_step(**kwargs)
            self.steps += 1

        log.info("Training finished.")

    def update_weights(self, state):
        """ Update the weight parameters of the network """
        pass

    def calculate_gradient(self, states, inputs, targets):
        """ Calculate the gradient for the network weights """
        pass


class FullBatchTrainer(Trainer):
    '''
    Full Batch Trainer.
    '''
    
    def __init__(self, *args, **kwargs):
        super(FullBatchTrainer, self).__init__(*args, **kwargs)
        log.info("FullBatchTrainer constructor")


class SGDTrainer(Trainer):
    '''
    Stochastic Gradient Descent trainer
    '''
    
    def get_training_patterns(self):
        inputs, targets = self.training_data
        self.context.randint(self.indices, 0, self.get_num_training_patterns())
        # copy minibatch
        self.context.copy_minibatch(inputs, self.indices, self.minibatch_inputs)
        self.context.copy_minibatch(targets, self.indices, self.minibatch_targets)

        return self.minibatch_inputs, self.minibatch_targets

    def get_batch_size(self):
        return self.minibatch_size

    def __init__(self, *args, **kwargs):
        self.minibatch_size = kwargs.get('minibatch_size', 128)
        super(SGDTrainer, self).__init__(*args, **kwargs)
        log.info("SGDTrainer constructor")


        thread = self.context.thread

        input_shape = self.network.shape[0]

        self.indices = thread.array((self.minibatch_size,), dtype=numpy.int32)
        self.minibatch_inputs = thread.array((self.minibatch_size,)+input_shape, dtype=numpy.float32)
        self.minibatch_targets = thread.array((self.minibatch_size,)+self.network.get_target_shape(), dtype=self.network.targets_dtype)