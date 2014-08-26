from neuro import reshape
from neuro.model import LayerState
import numpy
import logging


log = logging.getLogger("dense")

class DenseLayerState(LayerState):
    pass

class DenseLayer(object):

    def __init__(self, context, input_shape, num_units=128, **kwargs):
        log.info("DenseLayer constructor")
        self.context = context
        thr = context.thread

        if not isinstance(input_shape, tuple):
            input_shape = (input_shape,)
        if len(input_shape) > 1:
            d = 1
            for dim in input_shape:
                d *= dim
            input_shape = (d,)

        self.input_shape = input_shape
        self.output_shape = (num_units,)

        weights_shape = input_shape + self.output_shape

        weights = thr.array(weights_shape, dtype=numpy.float32)
        bias = thr.array((num_units,), dtype=numpy.float32)

        self.weights = weights
        self.bias = bias

        self.targets_shape = self.output_shape

    def create_state(self, n, state=None):
        log.info("DenseLayer create_state")
        thread = self.context.thread
        activation_shape = (n,) + self.output_shape
        if state is None:
            state = DenseLayerState(activation_shape)
        state.activations = thread.array(activation_shape, numpy.float32)
        return state

    def initialize_test_state(self, state):
        log.info("DenseLayer initialize_test_state")
        thread = self.context.thread
        state.deltas = thread.array(state.activations.shape, numpy.float32)
        return state

    def initialize_training_state(self, state):
        log.info("DenseLayer initialize_training_state")
        thread = self.context.thread
        state.deltas = thread.array(state.activations.shape, numpy.float32)
        state.gradients = thread.array(self.weights.shape, numpy.float32)
        state.gradients_bias = thread.array(self.bias.shape, numpy.float32)
        return state

    def reset(self, std=0.01):
        log.info("DenseLayer.reset")
        self.context.normal(self.weights, 0.0, std)
        self.bias.fill(numpy.float(0.0))

    def propagate(self, state, inputs):
        desired_shape = inputs.shape[:1] + self.input_shape
        inputs = reshape(inputs, desired_shape)
        self.context.dot(inputs, self.weights, state.activations)

    def transfer(self, state):
        pass

    def derivative(self, state):
        pass

    def calculate_gradient(self, state, prev_activations):
        delta = state.deltas
        gradient_weights, gradient_bias = state.gradients, state.gradients_bias

        desired_shape = prev_activations.shape[:1] + self.input_shape
        prev_activations = reshape(prev_activations, desired_shape)

        ctx = self.context
        # calculate the gradients
        ctx.dot(prev_activations, delta, gradient_weights, trans_a=True)
        # bias gradient is just the sum of the deltas
        # (because bias activation is implicitly 1.0)
        ctx.sum(delta, gradient_bias, axis=0)

    def backpropagate(self, state, prev_delta):
        deltas = state.deltas
        weights = self.weights

        desired_shape = prev_delta.shape[:1] + self.input_shape
        prev_delta = reshape(prev_delta, desired_shape)
        ctx = self.context
        ctx.dot(deltas, weights, prev_delta, trans_b=True)