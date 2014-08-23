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
        convolve(self.context, activations, self.weights, next_activations)
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

def get_output_shape(array1, array2, mode):
    """ Returns the shape of the intermediate array and the expected output array for the given arrays and mode """

    output_shape = None

    if mode == 'propagation':
        n, channels_1, width, height = array1.shape # inputs
        channels, filters, f_width, f_height = array2.shape # weights
        assert channels == channels_1

        out_0 = width - f_width + 1
        out_1 = height - f_height + 1
        output_shape = (n, channels, filters, out_0, out_1)

    if mode == 'backprop':
        n, filters_1, width, height = array1.shape # deltas
        channels, filters, f_width, f_height = array2.shape # weights
        assert filters_1 == filters

        out_0 = width + f_width - 1
        out_1 = height + f_height - 1
        output_shape = (n, channels, filters, out_0, out_1)

    if mode == 'gradient':
        n_1, channels, width, height = array1.shape # prev_deltas
        n, filters, f_width, f_height = array2.shape # deltas
        assert n_1 == n

        out_0 = width - f_width + 1
        out_1 = height - f_height + 1
        output_shape = (n, channels, filters, out_0, out_1)

    if output_shape is None:
        raise Exception("Invalid mode:", str(mode))

    return output_shape


def convolve2d_full(ctx, array, weights, dest):
    """ The output is the full discrete linear convolution of the inputs. """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (convolve2d_full, array.shape, weights.shape, thread)
    if not key in kernel_cache.keys():
        logging.info("compiling " + str(key))

        channels, filters, owidth, oheight = weights.shape[0], weights.shape[1], dest.shape[1], dest.shape[2]

        render_kwds = {
            'w0': weights.shape[2],
            'w1': weights.shape[3],
            'a0': array.shape[2],
            'a1': array.shape[3]
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
        // dest (filters, owidth, oheight)

        float a = 0.0f;
        SIZE_T x, y, i, j;
        const SIZE_T number = ${idxs[0]};
        const SIZE_T channel = ${idxs[1]};
        const SIZE_T filter = ${idxs[2]};
        const SIZE_T xout = ${idxs[3]};
        const SIZE_T yout = ${idxs[4]};
        for (i=0; i < ${w0}; i++){
            for (j=0; j < ${w1}; j++){
                x = xout - i;
                if (x < 0) continue;
                if (x >= ${a0}) continue;
                y = yout - j;
                if (y < 0) continue;
                if (y >= ${a1}) continue;
                a += ${array.load_idx}(number, channel, x,y) // filters, x, y
                   * ${weights.load_idx}(channel, filter, i, j); // channel, filter, i, j
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

        channels, filters, owidth, oheight = weights.shape[0], weights.shape[1], dest.shape[1], dest.shape[2]

        render_kwds = {
            'w0': weights.shape[2],
            'w1': weights.shape[3],
            'a0': array.shape[2],
            'a1': array.shape[3],
            'off0': int(weights.shape[2] - 1),
            'off1': int(weights.shape[3] - 1)
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
        const SIZE_T number = ${idxs[0]};
        const SIZE_T channel = ${idxs[1]};
        const SIZE_T filter = ${idxs[2]};
        const SIZE_T xout = ${idxs[3]};
        const SIZE_T yout = ${idxs[4]};
        for (i=0; i < ${w0}; i++){
            for (j=0; j < ${w1}; j++){
                x = xout - i  + ${off0};
                y = yout - j  + ${off1};
                a += ${array.load_idx}(number, channel, x, y)
                   * ${weights.load_idx}(channel, filter, i, j); // channel, filter, i, j
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

        channels, filters, owidth, oheight = weights.shape[0], weights.shape[1], dest.shape[1], dest.shape[2]

        render_kwds = {
            'w0': weights.shape[2],
            'w1': weights.shape[3],
            'a0': array.shape[2],
            'a1': array.shape[3],
            'off0': int(numpy.ceil(weights.shape[2] / 2.0) - 1),
            'off1': int(numpy.ceil(weights.shape[3] / 2.0) - 1),
            'c' : array.shape[1]
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
        const SIZE_T number = ${idxs[0]};
        const SIZE_T channel = ${idxs[1]};
        const SIZE_T filter = ${idxs[2]};
        const SIZE_T xout = ${idxs[3]};
        const SIZE_T yout = ${idxs[4]};
        for (i=0; i < ${w0}; i++){
            for (j=0; j < ${w1}; j++){
                x = xout - i  + ${off0};
                y = yout - j  + ${off1};
                if (0 <= x && x < ${a0} && 0 <= y && y < ${a1}) {
                    a += ${array.load_idx}(number, channel, x, y)
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


def convolve_backprop(ctx, deltas, weights, prev_deltas):
    """ convolve the deltas over the inputs. similar to full convolution but without shared weights """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (convolve_backprop, deltas.shape, weights.shape, thread)
    if not key in kernel_cache.keys():
        # deltas = (n, channels, width, height)
        # weights = (channels, filters, fwidth, fheight)
        # prev_deltas = (channels, iwidth, iheight)

        logging.info("compiling" + str(key))

        (n, channels, width, height) = deltas.shape
        (channels, filters, fwidth, fheight) = weights.shape


        render_kwds = {
            'filters':filters,
            'channels': channels,
            'width': width,
            'height': height,
            'fwidth': fwidth,
            'fheight': fheight,
        }

        kernel_conv = PureParallel(
            [
                Parameter('deltas', Annotation(deltas, 'i')),
                Parameter('weights', Annotation(weights, 'i')),
                Parameter('prev_deltas', Annotation(prev_deltas, 'o'))
            ],
            """
        float a = 0.0f;
        SIZE_T x, y, i, j;
        const SIZE_T number = ${idxs[0]};
        const SIZE_T channel = ${idxs[1]};
        const SIZE_T filter = ${idxs[2]};
        const SIZE_T xout = ${idxs[3]};
        const SIZE_T yout = ${idxs[4]};
        for (i=0; i < ${w0}; i++){
            for (j=0; j < ${w1}; j++){
                x = xout - i;
                if (x < 0) continue;
                if (x >= ${a0}) continue;
                y = yout - j;
                if (y < 0) continue;
                if (y >= ${a1}) continue;
                a += ${array.load_idx}(number, channel, x,y) // filters, x, y
                   * ${weights.load_idx}(channel, filter, i, j); // channel, filter, i, j
            }
        }

        ${dest.store_same}(a);

        """, guiding_array='dest', render_kwds=render_kwds)

        kernel_cache[key] = kernel_conv.compile(
            thread, fast_math=True)

    # run convolution -> intermediate
    kernel_cache[key](array, weights, dest)

    return dest