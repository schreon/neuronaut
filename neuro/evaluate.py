import cPickle
import logging
import os

import numpy

log = logging.getLogger("evaluate")
class ConfigurationEvaluator():
    '''
    Evaluates a neural network configuration.
    '''
    
    def __init__(self, **kwargs):
        
        self.NetworkClass = kwargs.get("NetworkClass")  
        self.TrainerClass = kwargs.get("TrainerClass")        
        self.network_args = kwargs.get("network_args")
        self.network_structure = kwargs.get("network_structure")
        self.trainer_args = kwargs.get("trainer_args")
        
        self.data_train = kwargs.get("data_train")
        self.data_test = kwargs.get("data_test")

    def optimize(self):
        '''
        Performs one training session using the defined neural network configuration.
        '''
        # build the network                
        network = self.NetworkClass(input_shape=self.data_train[0].shape[1], **self.network_args)
        # skip the first entry because its the input layer        
        for options in self.network_structure[1:]:
            network.add_layer(options['LayerClass'], options['num_units'])
        network.reset_weights(std=0.01)
        
        # upload data
        ctx = network.context        
        inputs, targets = self.data_train
        inputs_test, targets_test = self.data_test        
        inputs, targets, inputs_test, targets_test = ctx.upload(inputs, targets, inputs_test, targets_test)
        
        # build the trainer
        trainer = self.TrainerClass(network=network, **self.trainer_args)
        training_size = self.trainer_args.get("minibatch_size", inputs.shape[0])
        training_state = trainer.TrainingState(network=network, size=training_size)
        test_state = trainer.TestState(network=network, size=inputs_test.shape[0])
        
        # do the training
        trainer.train(training_state, test_state, inputs, targets, inputs_test, targets_test)
        
        # download weights as numpy arrays
        weights = network.download_weights()
        
        # return the weights and the error history
        return weights, trainer.errors
    
    def evaluate(self, num_evaluations=10, file_name=None):
        '''
        Repeatedly performs training sessions and captures the results.
        
        :param num_evaluations: number of training sessions
        :param file_name: file to store the results (optional)
        '''
        # continue if the file already exists.
        if file_name is not None and os.path.exists(file_name):
            with open(file_name, "rb") as f:
                (results, test_errors) = cPickle.load(f)
            log.info("Continue with existing evaluation %s" % file_name)
        else:
            log.info("New evaluation session.")
            results = []
            test_errors = []
        
        best_weights = None
        for _ in xrange(num_evaluations):
            weights, errors = self.optimize()
            results.append((weights, errors))
            # new best?
            if len(test_errors) < 1 or errors['best']['test'] < test_errors[-1]:
                best_weights = weights
            test_errors.append(errors['best']['test'])
            test_std = numpy.std(test_errors)
            test_mean = numpy.mean(test_errors)
            log.info("test error mean: %.4f, test error std: %.4f" % (test_mean, test_std))
            
            if file_name is not None:
                with open(file_name, "wb") as f:
                    cPickle.dump((results, test_errors), f, protocol=-1)
        return best_weights