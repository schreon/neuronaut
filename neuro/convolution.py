import logging
from neuro import reshape
from neuro.model import LayerState

import numpy
from reikna.algorithms import PureParallel
from reikna.core import Parameter
from reikna.core.signature import Annotation

log = logging.getLogger("convolution")

class Convolution2DState(LayerState):
    pass

class Convolution2DLayer(object):
    
    def __init__(self, context, input_shape, filter_shape=(32, 4, 4), **kwargs):
        log.info("ConvolutionLayer constructor")
        self.context = context
        thr = context.thread

        number_channels, height_channels, width_channels = input_shape
        number_filters, height_filters, width_filters = filter_shape

        weights = numpy.random.randn(number_channels, number_filters, height_filters, width_filters).astype(numpy.float32)
        bias = numpy.random.randn(number_filters).astype(numpy.float32)

        self.weights, self.bias = context.upload(weights, bias)

        self.input_shape = input_shape
        self.output_shape = get_output_shape(input_shape, self.weights, 'propagation')[1:]

    def create_state(self, num_patterns, state=None):
        # n, width, height = shape
        activation_shape = (num_patterns,) + get_output_shape(self.input_shape, self.weights, 'propagation')
        activations_intermediate = numpy.zeros(activation_shape).astype(numpy.float32)
        activations = numpy.sum(activations_intermediate, axis=1)
        if state is None:
            state = Convolution2DState(activations.shape)
        state.activations_intermediate, state.activations = self.context.upload(activations_intermediate, activations)
        return state

    def initialize_training_state(self, state):
        log.info("Convolution2DLayer initialize_training_state")
        thread = self.context.thread
        state.deltas = thread.array(state.activations.shape, numpy.float32)
        state.gradients = thread.array(self.weights.shape, numpy.float32)
        state.gradients_bias = thread.array(self.bias.shape, numpy.float32)
        return state

    def create_training_state(self, input_shape, state):
        act_shape = get_output_shape(input_shape, self.weights, 'propagation')
        act_shape[:1] + act_shape[2:]
        deltas = numpy.zeros(act_shape)
        deltas_intermediate = numpy.zeros(get_output_shape(deltas, self.weights, 'backprop')).astype(numpy.float32)
        prev_deltas = deltas_intermediate.sum(axis=2)
        gradient_intermediate = numpy.zeros(get_output_shape(prev_deltas, deltas, 'gradient'))
        gradient = gradient_intermediate.sum(axis=0).sum(axis=0).sum(axis=0)

        state.deltas = state.context.upload(deltas)
        state.deltas_intermediate = state.context.upload(deltas_intermediate)
        state.gradient_intermediate = state.context.upload(gradient_intermediate)
        state.gradient = state.context.upload(gradient)

    def propagate(self, state, inputs, **kwargs):
        convolve2d_propagation(self.context, inputs, self.weights, state.activations)

    def transfer(self, activations):
        # TODO: apply transfer function
        self.context.add(state.activations, self.bias)
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

def get_output_shape(shape1, shape2, mode):
    """ Returns the shape of the intermediate array and the expected output array for the given arrays and mode """

    if not isinstance(shape1, tuple):
        shape1 = shape1.shape
    if not isinstance(shape2, tuple):
        shape2 = shape2.shape

    output_shape = None

    if mode == 'propagation':
        channels_1, width, height = shape1 # inputs
        channels, filters, f_width, f_height = shape2 # weights
        assert channels == channels_1

        out_0 = width - f_width + 1
        out_1 = height - f_height + 1
        output_shape = (channels, filters, out_0, out_1)

    if mode == 'backprop':
        n, filters_1, width, height = shape1 # deltas
        channels, filters, f_width, f_height = shape2 # weights
        assert filters_1 == filters

        out_0 = width + f_width - 1
        out_1 = height + f_height - 1
        output_shape = (channels, filters, out_0, out_1)

    if mode == 'gradient':
        n_1, channels, p_width, p_height = shape1 # prev_deltas
        n, filters, d_width, d_height = shape2 # deltas
        assert n_1 == n
        out_0 = p_width - d_width + 1
        out_1 = p_height - d_height + 1
        output_shape = (d_width, d_height, channels, filters, out_0, out_1)

    if output_shape is None:
        raise Exception("Invalid mode:", str(mode))

    return output_shape


def convolve2d_propagation(ctx, array, weights, dest):
    """ The output is the valid discrete linear convolution of the inputs. """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (convolve2d_propagation, weights.shape, array.shape, thread)
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



def convolve2d_backprop(ctx, deltas, weights, deltas_intermediate):
    """ The output is the full discrete linear convolution of the inputs. """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (convolve2d_backprop, deltas.shape, weights.shape, thread)
    if not key in kernel_cache.keys():
        logging.info("compiling " + str(key))

        # Extract shapes from the arrays
        channels, filters, f_width, f_height = weights.shape
        n_1, filters_1, d_width, d_height = deltas.shape
        n, channels_1, filters_2, p_width, p_height = deltas_intermediate.shape

        # Some assertions to be sure everything is correct
        assert n_1 == n
        assert filters_2 == filters_1 == filters
        assert channels_1 == channels
        expected_shape = get_output_shape(deltas, weights, 'backprop')
        assert expected_shape == deltas_intermediate.shape

        # Render keywords
        render_kwds = {
            'n':n,
            'filters':filters,
            'channels': channels,
            'f_width': f_width,
            'f_height': f_height,
            'd_width': d_width,
            'd_height': d_height,
            'p_width': p_width,
            'p_height': p_height,
        }

        # The kernel
        kernel = PureParallel(
            [
                Parameter('deltas', Annotation(deltas, 'i')),
                Parameter('weights', Annotation(weights, 'i')),
                Parameter('deltas_intermediate', Annotation(deltas_intermediate, 'o'))
            ],
            """
        float d = 0.0f;
        SIZE_T x, y, i, j, fi, fj;
        const SIZE_T number = ${idxs[0]};
        const SIZE_T channel = ${idxs[1]};
        const SIZE_T filter = ${idxs[2]};
        const SIZE_T xout = ${idxs[3]};
        const SIZE_T yout = ${idxs[4]};
        for (i=0; i < ${f_width}; i++){
            for (j=0; j < ${f_height}; j++){
                x = xout - i;
                if (x < 0) continue;
                if (x >= ${d_width}) continue;
                y = yout - j;
                if (y < 0) continue;
                if (y >= ${d_height}) continue;
                // acces weights in flipped order!
                fi = ${f_width} - i - 1;
                fj = ${f_height} - j - 1;
                d += ${deltas.load_idx}(number, channel, x, y)
                   * ${weights.load_idx}(channel, filter, fi, fj);
            }
        }

        ${deltas_intermediate.store_same}(d);

        """, guiding_array='deltas_intermediate', render_kwds=render_kwds)

        kernel_cache[key] = kernel.compile(
            thread, fast_math=True)

    # run convolution -> intermediate
    kernel_cache[key](deltas, weights, deltas_intermediate)

    return deltas_intermediate

def convolve2d_gradient(ctx, prev_deltas, deltas, gradient_intermediate):
    """ The output is the full discrete linear convolution of the inputs. """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    key = (convolve2d_gradient, prev_deltas.shape, deltas.shape, thread)
    if not key in kernel_cache.keys():
        logging.info("compiling " + str(key))

        # Extract shapes from the arrays
        n, channels, p_width, p_height = prev_deltas.shape
        n_1, filters, d_width, d_height = deltas.shape
        n, d_width_1, d_height_1, channels_1, filters_1, f_width, f_height = gradient_intermediate.shape

        # Some assertions to be sure everything is correct
        assert n_1 == n
        assert filters_1 == filters
        assert channels_1 == channels
        expected_shape = get_output_shape(prev_deltas, deltas, 'gradient')
        assert expected_shape == gradient_intermediate.shape
        assert d_width_1 == d_width
        assert d_height_1 == d_height

        # Render keywords
        render_kwds = {
            'n':n,
            'filters':filters,
            'channels': channels,
            'f_width': f_width,
            'f_height': f_height,
            'd_width': d_width,
            'd_height': d_height,
            'p_width': p_width,
            'p_height': p_height,
        }

        # The kernel
        kernel = PureParallel(
            [
                Parameter('prev_deltas', Annotation(prev_deltas, 'i')),
                Parameter('deltas', Annotation(deltas, 'i')),
                Parameter('gradient_intermediate', Annotation(gradient_intermediate, 'o'))
            ],
            """

        const SIZE_T number = ${idxs[0]};
        const SIZE_T dx = ${idxs[1]};
        const SIZE_T dy = ${idxs[2]};
        const SIZE_T channel = ${idxs[3]};
        const SIZE_T filter = ${idxs[4]};
        const SIZE_T fx = ${idxs[5]};
        const SIZE_T fy = ${idxs[6]};


        // weight gradient at the weight position fx, fy is defined by the sum
        //
        //       (deltas * prev_deltas[fx:d_width+fx, fy:fy+d_height]).sum()
        //
        // alternatively we can store all delta positions and sum in a separate kernel - this is what we do now.

        float g = ${deltas.load_idx}(number, filter, dx, dy) * ${prev_deltas.load_idx}(number, channel, dx+fx, dy+fy);

        ${gradient_intermediate.store_same}(g);

        """, guiding_array='gradient_intermediate', render_kwds=render_kwds)

        kernel_cache[key] = kernel.compile(
            thread, fast_math=True)

    # run convolution -> intermediate
    kernel_cache[key](prev_deltas, deltas, gradient_intermediate)

    return gradient_intermediate
