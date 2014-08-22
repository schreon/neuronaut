import logging
from neuro import reshape
import neuro

import numpy
from reikna.algorithms import PureParallel
from reikna.core import Parameter
from reikna.core.signature import Annotation

log = logging.getLogger("classification")


def classification_delta_kernel(ctx, outputs, targets, deltas):
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    assert outputs.shape[0] == targets.shape[0] == deltas.shape[0]
    assert len(targets.shape) == 1
    assert targets.dtype == numpy.int32
    assert outputs.shape[1] == deltas.shape[1]


    key = (classification_delta_kernel, outputs.shape)
    if not key in kernel_cache.keys():
        log.info("compiling " + str(key))
        kernel = PureParallel(
            [
                Parameter('outputs', Annotation(outputs, 'i')),
                Parameter('targets', Annotation(targets, 'i')),
                Parameter('deltas', Annotation(deltas, 'o'))
            ],
        """
        ${outputs.ctype} out = ${outputs.load_same};
        SIZE_T t = ${targets.load_idx}(${idxs[0]});
        SIZE_T idx = ${idxs[1]};
        ${deltas.ctype} d;
        if (t == idx) {
            d = 1.0f - out;
        } else {
            d = -out;
        }
        ${deltas.store_same}(d);
        """, guiding_array='deltas')

        kernel_cache[key] = kernel.compile(thread)

    # Run kernel
    kernel_cache[key](outputs, targets, deltas)

def class_errors(ctx, expected, actual, errors):
    """ expected int32, actual float, errors int32 """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (class_errors, expected.shape)

    if key not in kernel_cache.keys():
        # target should be an integer
        logging.info("compiling " + str(key))
        assert expected.shape == errors.shape # one neuron per class
        assert expected.shape == (actual.shape[0],) # index of the class
        assert actual.dtype == numpy.float32
        assert expected.dtype == numpy.int32
        assert errors.dtype == numpy.int32
        kernel = PureParallel(
            [
                Parameter('expected', Annotation(expected, 'i')),
                Parameter('actual', Annotation(actual, 'i')),
                Parameter('errors', Annotation(errors, 'o'))
            ],
            """
            SIZE_T expected = ${expected.load_idx}(${idxs[0]});;
            float maximum=0.0f;
            float value;
            SIZE_T maxindex = 0;

            SIZE_T tl = ${target_length};

            // calculate argmax
            for(SIZE_T j=0; j < tl; j++) {
                value = ${actual.load_idx}(${idxs[0]}, j);

                if (value > maximum) {
                    maximum = value;
                    maxindex = j;
                }
            }

            // If the confidence is too low, return an error
            if (maximum < (1.0f / ${target_length}.0f + 0.001f)) {
                ${errors.store_same}(1);
                return;
            };

            // compare argmax
            if (maxindex != expected) {
                ${errors.store_same}(1);
            } else {
                ${errors.store_same}(0);
            }

        """, guiding_array='expected', render_kwds={'target_length' : numpy.int32(actual.shape[1])})

        kernel_cache[key] = kernel.compile(thread)

    kernel_cache[key](expected, actual, errors)

class Classification(object):
    """
    Defines the ouput of a neural network to solve a regression task.
    """

    def __init__(self, **kwargs):        
        super(Classification, self).__init__(**kwargs)
        log.info("Classification constructor")

        self.targets_dtype = numpy.int32
        self.error_measure = "Classification Errors"
    
    def delta(self, state, targets):
        """
        Classes must be coded as integers. Each integer is one class.
        """
        super(Classification, self).delta(state, targets)
        classification_delta_kernel(self.context, state.activations[-1], targets, state.deltas[-1])

    def add_layer(self, **kwargs):
        super(Classification, self).add_layer(**kwargs)

        # in a classification network, the target values are just the index of the correct class
        self.layers[-1].targets_shape = ()

    def error(self, inputs, targets, state):
        """
        Calculate the classification error.
        """
        self.propagate(state, inputs)
        class_errors(self.context, targets, state.activations[-1], state.classification_errors)
        self.context.sum(state.classification_errors, state.error)

        return state.error.get()

class ClassificationState(object):
    '''
    Holds the state belonging to classification networks.
    '''

    def __init__(self, **kwargs):
        super(ClassificationState, self).__init__(**kwargs)
        ctx = kwargs['network'].context

        shp = (kwargs['size'],)
        self.classification_errors = ctx.thread.array(shp, dtype=numpy.int32)
        log.info(self.classification_errors.shape)

class ClassificationTrainer(object):
    def __init__(self, **kwargs):
        super(ClassificationTrainer, self).__init__(**kwargs)
        log.info("ClassificationTrainer constructor")

        self.TrainingState = neuro.create("TrainingState", self.TrainingState, ClassificationState)
        self.TestState = neuro.create("TestState", self.TestState, ClassificationState)