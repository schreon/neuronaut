import logging
from reikna.algorithms import PureParallel
from reikna.core import Parameter
from reikna.core.signature import Annotation

log = logging.getLogger("linear")


def linear(context, activations, bias, dest=None):
    kernel_cache, thread = context.kernel_cache, context.thread

    if dest is None:
        dest = activations

    key = (linear, activations.shape, thread)
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

        ${dest.store_same}(a);
        """, guiding_array='activations')

        kernel_cache[key] = kernel.compile(thread, fast_math=True)

    # Run kernel
    kernel_cache[key](activations, bias, dest)

    return dest

class LinearLayer(object):
    def __init__(self, context, *args, **kwargs):
        log.info("LinearLayer constructor")
        super(LinearLayer, self).__init__(context, *args, **kwargs)

    def transfer(self, state):
        linear(self.context, state.activations, self.bias)

    def derivative(self, state):
        # no need to do anything
        pass
