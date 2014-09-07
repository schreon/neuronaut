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

class ClassificationNetwork(object):
    """
    Defines the ouput of a neural network to solve a regression task.
    """

    def __init__(self, **kwargs):        
        super(ClassificationNetwork, self).__init__(**kwargs)
        log.info("Classification constructor")

        self.targets_dtype = numpy.int32
        self.error_measure = "Classification Errors"

    def create_state(self, num_patterns):
        state = super(ClassificationNetwork, self).create_state(num_patterns)
        ctx = self.context
        shp = (num_patterns,)
        state.classification_errors = ctx.thread.array(shp, dtype=numpy.int32)
        return state

    def get_target_shape(self):
        # in a classification network, the target values are just the index of the correct class
        return ()

    def get_target_dtype(self):
        return numpy.int32

    def delta(self, network_state, targets):
        """
        Classes must be coded as integers. Each integer is one class.
        """
        super(ClassificationNetwork, self).delta(network_state, targets)
        outputs = network_state.layers[-1].activations
        deltas = network_state.layers[-1].deltas
        classification_delta_kernel(self.context, outputs, targets, deltas)

    def add_layer(self, LayerClass,  **kwargs):
        super(ClassificationNetwork, self).add_layer(LayerClass, **kwargs)

    def error(self, inputs, targets, network_state):
        """
        Calculate the classification error.
        """
        self.propagate(network_state, inputs)
        outputs = network_state.layers[-1].activations

        class_errors(self.context, targets, outputs, network_state.classification_errors)
        self.context.sum(network_state.classification_errors, network_state.error)

        return network_state.error.get()
