import neuro
from neuro.backpropagation import Backpropagation
from neuro.evaluate import ConfigurationEvaluator
from neuro.model import FeedForwardNeuralNetwork, Regression, NaNMask
from neuro.rprop import RPROP
from neuro.stopping import EarlyStopping
from neuro.training import FullBatchTrainer
from neuro.weightdecay import Renormalize


class RegressionNetworkFactory(object):
    """
    Trains a neural network with logistic hidden units and a linear regression output layer.
    Uses FullBatch, RPROP, NaNMask, Renormalize for training.
    """
    def __init__(self, context):
        self.context = context
        self.structure = []

    def add_layer(self, num_units):
        self.structure.append(num_units)
    
    def train(self, data, validation_ratio=0.3, retries=5, file_name=None):  
        ctx = self.context
         
        (inp, targ, mi, ma) = data      
        n = int(validation_ratio*inp.shape[0])
        inp_test = inp[:n]
        targ_test = targ[:n]        
        inp = inp[n:]
        targ = targ[n:]        
        data_train = (inp, targ)
        data_test = (inp_test, targ_test)

        NetworkClass = neuro.create("MyNetwork",
                                 FeedForwardNeuralNetwork,
                                 Regression,
                                 NaNMask,
                                 )

        network_structure = [dict(num_units=inp.shape[1])]
        
        for dim in self.structure:
            network_structure.append(dict(num_units=dim, function=ctx.logistic, derivative=ctx.logistic_derivative))
        
        network_structure.append(dict(num_units=targ.shape[1], function=ctx.linear, derivative=ctx.linear_derivative))
        
        TrainerClass = neuro.create("MyTrainer", 
                                 FullBatchTrainer, 
                                 Backpropagation, 
                                 EarlyStopping, 
                                 RPROP, 
                                 Renormalize
                                 )
        
        trainer_options = dict(min_steps=1000,
                            validation_frequency=5, # validate the model every 5 steps
                            validate_train=False, # ado not validate on training set
                            logging_frequency=2.0, # log progress every 2 seconds
                            max_weight_size=15.0,
                            max_step_size=1.0
                            )

        evaluator = ConfigurationEvaluator(
          data_train=data_train,
          data_test=data_test,
          NetworkClass=NetworkClass,
          TrainerClass=TrainerClass,
          network_args=dict(context=ctx),
          trainer_args=trainer_options,
          network_structure=network_structure
          )

        weights = evaluator.evaluate(num_evaluations=retries, file_name=file_name)
        
        return NetworkClass, weights
        