import logging
from neuro import reshape

import numpy
from reikna.algorithms import PureParallel
from reikna.core import Parameter
from reikna.core.signature import Annotation

log = logging.getLogger("convolution")

class Convolution2DLayer(object):
    
    def __init__(self, context, input_shape, num_units=16, filter_shape=(3,3), **kwargs):
        log.info("ConvolutionLayer constructor")
        self.context = context
        thr = context.thread
        
        d1, d2 = input_shape
        f1, f2 = filter_shape
        d1 = d1 - d1 % f1
        d2 = d2 - d2 % f2
        
        self.input_shape = input_shape
        self.output_shape = (d1, d2)
        
        
        weights_shape = (num_units,) + filter_shape
        
        weights = thr.array(weights_shape, dtype=numpy.float32)
        bias = thr.array((num_units,), dtype=numpy.float32)       
         
        self.weights = weights
        self.bias = bias
    
    def propagate(self, activations, next_activations):
        # TODO: convolve
        pass

    def transfer(self, activations):
        # TODO: apply transfer function
        pass

    def derivative(self, activations, delta):
        # TODO: apply derivative of the transfer function
        pass
    
    def calculate_gradient(self, prev_activations, delta, gradient_weights, gradient_bias):
        # TODO: convolve deltas over input activations
        pass
    
    def backpropagate(self, delta, weights, prev_delta):
        # TODO: convolve weights over deltas
        pass

def get_output_shape(array, weights, mode):
    """ Returns the shape of the intermediate array and the expected output array for the given arrays and mode """

    shape = None
    if mode == 'full':
        out_0 = array.shape[1] + weights.shape[2] - 1
        out_1 = array.shape[2] + weights.shape[3] - 1
        shape = (out_0, out_1)

    if mode == 'valid':
        out_0 = array.shape[1] - weights.shape[2] + 1
        out_1 = array.shape[2] - weights.shape[3] + 1
        shape = (out_0, out_1)

    if mode == 'same':
        out_0 = array.shape[1]
        out_1 = array.shape[2]
        shape = (out_0, out_1)

    if shape is None:
        raise Exception("Invalid mode:", str(mode))

    (channels, filters, _, _,) = weights.shape

    output_shape = (channels, filters, shape[0], shape[1])

    return output_shape


def convolve2d_full(ctx, array, weights, dest):
    """ The output is the full discrete linear convolution of the inputs. """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (convolve2d_full, array.shape, weights.shape, thread)
    if not key in kernel_cache.keys():
        logging.info("compiling " + str(key))

        render_kwds = {
            'w0': weights.shape[2],
            'w1': weights.shape[3],
            'a0': array.shape[1],
            'a1': array.shape[2],
        }

        kernel_conv = PureParallel(
            [
                Parameter('array', Annotation(array, 'i')),
                Parameter('weights', Annotation(weights, 'i')),
                Parameter('dest', Annotation(dest, 'o'))
            ],
            """
        // Array dimensions:
        // array : (channels, width, height)
        // weights: (channels, filters, fwidth, fheight)
        // dest (channels, filters, owidth, oheight)

        float a = 0.0f;
        SIZE_T x, y, i, j;
        SIZE_T channel = ${idxs[0]};
        SIZE_T filter = ${idxs[1]};
        for (i=0; i < ${w0}; i++){
            for (j=0; j < ${w1}; j++){
                x = ${idxs[2]} - i;
                y = ${idxs[3]} - j;
                if (0 <= x && x < ${a0} && 0 <= y && y < ${a1}) {
                    a += ${array.load_idx}(channel, x,y) // filters, x, y
                       * ${weights.load_idx}(channel, filter, i, j); // channel, filter, i, j
                }
            }
        }
        ${dest.store_same}(a);
        """, guiding_array='dest', render_kwds=render_kwds)
        kernel_cache[key] = kernel_conv.compile(
            thread, fast_math=True)

    # run convolution -> intermediate
    kernel_cache[key](array, weights, dest)

    return dest


def convolve2d_valid(ctx, array, weights, dest):
    """ The output is the valid discrete linear convolution of the inputs. """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (convolve2d_valid, weights.shape, array.shape, thread)
    if not key in kernel_cache.keys():
        logging.info("compiling" + str(key))

        render_kwds = {
            'w0': weights.shape[2],
            'w1': weights.shape[3],
            'a0': array.shape[1],
            'a1': array.shape[2],
            'off0': int(weights.shape[2] - 1),
            'off1': int(weights.shape[3] - 1),
        }

        kernel_conv = PureParallel(
            [
                Parameter('array', Annotation(array, 'i')),
                Parameter('weights', Annotation(weights, 'i')),
                Parameter('dest', Annotation(dest, 'o'))
            ],
            """
        // Array dimensions:
        // array : (channels, width, height)
        // weights: (channels, filters, fwidth, fheight)
        // dest (channels, filters, owidth, oheight)

        float a = 0.0f;
        SIZE_T x, y, i, j;
        SIZE_T channel = ${idxs[0]};
        SIZE_T filter = ${idxs[1]};
        for (i=0; i < ${w0}; i++){
            for (j=0; j < ${w1}; j++){
                x = ${idxs[2]} - i  + ${off0};
                y = ${idxs[3]} - j  + ${off1};
                if (0 <= x && x < ${a0} && 0 <= y && y < ${a1}) {
                    a += ${array.load_idx}(channel, x,y) // channel, x, y
                       * ${weights.load_idx}(channel, filter, i, j); // channel, filter, i, j
                }
            }
        }
        ${dest.store_same}(a);
        """, guiding_array='dest', render_kwds=render_kwds)
        kernel_cache[key] = kernel_conv.compile(
            thread, fast_math=True)

    # run convolution
    kernel_cache[key](array, weights, dest)

    return dest


def convolve2d_same(ctx, array, weights, dest):
    """ The output is the same size as array, centered with respect to the full output. """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (convolve2d_same, array.shape, weights.shape, thread)
    if not key in kernel_cache.keys():
        logging.info("compiling" + str(key))

        render_kwds = {
            'w0': weights.shape[2],
            'w1': weights.shape[3],
            'a0': array.shape[1],
            'a1': array.shape[2],
            'off0': int(numpy.ceil(weights.shape[2] / 2.0) - 1),
            'off1': int(numpy.ceil(weights.shape[3] / 2.0) - 1),
        }

        kernel_conv = PureParallel(
            [
                Parameter('array', Annotation(array, 'i')),
                Parameter('weights', Annotation(weights, 'i')),
                Parameter('dest', Annotation(dest, 'o'))
            ],
            """
        // Array dimensions:
        // array : (channels, width, height)
        // weights: (channels, filters, fwidth, fheight)
        // dest (channels, filters, owidth, oheight)

        float a = 0.0f;
        SIZE_T x, y, i, j;
        SIZE_T channel = ${idxs[0]};
        SIZE_T filter = ${idxs[1]};
        for (i=0; i < ${w0}; i++){
            for (j=0; j < ${w1}; j++){
                x = ${idxs[2]} - i  + ${off0};
                y = ${idxs[3]} - j  + ${off1};
                if (0 <= x && x < ${a0} && 0 <= y && y < ${a1}) {
                    a += ${array.load_idx}(channel, x,y) // channel, x, y
                       * ${weights.load_idx}(channel, filter, i, j); // channel, filter, i, j
                }
            }
        }
        ${dest.store_same}(a);
        """, guiding_array='dest', render_kwds=render_kwds)
        kernel_cache[key] = kernel_conv.compile(
            thread, fast_math=True)

    # run convolution
    kernel_cache[key](array, weights, dest)

    return dest

modes = {
    'full': convolve2d_full,
    'valid': convolve2d_valid,
    'same': convolve2d_same
}


def convolve(thr, array, weights, dest, mode):
    """ Convolve two arrays. Mimicks the behavior of scipy.signal.convolve2d """
    if not mode in modes.keys():
        raise Exception("invalid mode for convolve2d:" + str(mode))
    else:
        return modes[mode](thr, array, weights, dest)
