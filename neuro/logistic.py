import logging
from reikna.algorithms import PureParallel
from reikna.core import Parameter
from reikna.core.signature import Annotation

log = logging.getLogger("logistic")

def logistic(context, activations, bias, dest=None):
    kernel_cache, thread = context.kernel_cache, context.thread

    if dest is None:
        dest = activations

    key = (logistic, activations.shape, thread)
    if not key in kernel_cache.keys():
        log.info("compiling " + str(key))
        assert activations.shape[1] == bias.shape[0]

        kernel = PureParallel(
            [
                Parameter('activations', Annotation(activations, 'i')),
                Parameter('bias', Annotation(bias, 'i')),
                Parameter('dest', Annotation(dest, 'o')),
            ],
            """
        ${activations.ctype} a = ${activations.load_same};
        ${bias.ctype} b = ${bias.load_idx}(${idxs[1]});

        a += b;
        a = min(max(-45.0f, a), 45.0f);
        a = 1.0f / (1.0f + exp(-a));

        ${dest.store_same}(a);
        """, guiding_array='activations')

        kernel_cache[key] = kernel.compile(thread, fast_math=True)

    # Run kernel
    kernel_cache[key](activations, bias, dest)

    return dest

def logistic_derivative(context, activations, delta, dest=None):
    kernel_cache, thread = context.kernel_cache, context.thread

    if dest is None:
        dest = delta

    key = (logistic_derivative, activations.shape, thread)
    if not key in kernel_cache.keys():
        log.info("compiling " + str(key))
        kernel = PureParallel(
            [
                Parameter('activations', Annotation(activations, 'i')),
                Parameter('delta', Annotation(activations, 'i')),
                Parameter('dest', Annotation(dest, 'o')),
            ],
            """
        ${activations.ctype} a = ${activations.load_same};
        ${delta.ctype} d = ${delta.load_same};

        d = d*a*(1.0f - a);

        ${dest.store_same}(d);
        """, guiding_array='activations')

        kernel_cache[key] = kernel.compile(thread, fast_math=True)

    # Run kernel
    kernel_cache[key](activations, delta, dest)

class Logistic(object):
    def __init__(self, context, *args, **kwargs):
        log.info("Logistic constructor")
        super(Logistic, self).__init__(context, *args, **kwargs)

    def transfer(self, state):
        logistic(self.context, state.activations, self.bias)

    def derivative(self, state):
        logistic_derivative(self.context, state.activations, state.deltas)
