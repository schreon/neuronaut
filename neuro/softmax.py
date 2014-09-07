import logging
from neuro.logistic import logistic_derivative
import numpy
from reikna.algorithms import PureParallel
from reikna.core import Parameter
from reikna.core.signature import Annotation

log = logging.getLogger("softmax")

def softmax(ctx, activations, bias, dest=None):
    """ Softmax Activation Function """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    if dest is None:
        dest = activations

    key = (softmax, activations.shape)
    if key not in kernel_cache.keys():
        logging.info("compiling " + str(key))
        # Regression hidden layer
        kernel_softmax = PureParallel(
            [
                Parameter('activations', Annotation(activations, 'i')),
                Parameter('bias', Annotation(bias, 'i')),
                Parameter('dest', Annotation(dest, 'o')),
            ],
            """
            float x;
            float b;
            float s = 0.0f;
            SIZE_T tl = ${target_length};
            for(SIZE_T j=0; j < tl; j++) {
                x = ${activations.load_idx}(${idxs[0]}, j);
                b = ${bias.load_idx}(j);
                x += b;
                x = exp(min(max(x, -45.0f), 45.0f));
                ${dest.store_idx}(${idxs[0]}, j, x);

                s += x;
            }

            // divide by sum
            for(SIZE_T j=0; j < tl; j++) {
                x = ${dest.load_idx}(${idxs[0]}, j);
                x /= s;
                ${dest.store_idx}(${idxs[0]}, j, x);
            }
        """, guiding_array=(activations.shape[0],), render_kwds={'target_length' : numpy.int32(activations.shape[1])})

        kernel_cache[key] = kernel_softmax.compile(thread)

    kernel_cache[key](activations, bias, dest)

class Softmax(object):
    def __init__(self, context, *args, **kwargs):
        log.info("Softmax constructor")
        super(Softmax, self).__init__(context, *args, **kwargs)

    def transfer(self, state):
        softmax(self.context, state.activations, self.bias)

    def derivative(self, state):
        logistic_derivative(self.context, state.activations, state.deltas)
