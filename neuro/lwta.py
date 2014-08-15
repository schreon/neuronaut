import logging

import numpy
from reikna.algorithms.pureparallel import PureParallel
from reikna.core.signature import Parameter, Annotation

import neuro

log = logging.getLogger("lwta")

def lwta(ctx, mat, lwta_size):
    kernel_cache = ctx.kernel_cache
    lwta_size = numpy.float32(lwta_size)
    thread = ctx.thread
    key = (lwta, mat.dtype, mat.shape, lwta_size)

    if key not in kernel_cache.keys():
        num_units = mat.shape[1]
        log.info("compiling " + str(key))
        kernel = PureParallel(
            [
                Parameter('mat', Annotation(mat, 'io'))
            ],
            """
        SIZE_T this_idx = ${idxs[1]};
        SIZE_T group_size = ${lwta_size};
        // only the first thread per group computes anything
        if (this_idx % group_size == 0) {
            SIZE_T argmax = ${idxs[1]};
            SIZE_T candidate_idx;
            ${mat.ctype} ma = ${mat.load_same};
            ${mat.ctype} candidate_value;
            // find the argmax in the group
            for (SIZE_T i=1; i < group_size; i++) {
                candidate_idx = this_idx + i;
                if (candidate_idx >= ${num_units}) break;
                candidate_value = ${mat.load_idx}(${idxs[0]}, candidate_idx);
                if ( candidate_value > ma) {
                    ma = candidate_value;
                    argmax = candidate_idx;
                }
            }
            // second pass: zero all except argmax
            for (SIZE_T i=0; i < group_size; i++) {
                candidate_idx = this_idx + i;
                if (candidate_idx >= ${num_units}) break;
                if ( candidate_idx != argmax ) {
                    ${mat.store_idx}(${idxs[0]}, candidate_idx, 0.0f);
                }
            }
        }
            
        """, guiding_array='mat', render_kwds=dict(lwta_size=lwta_size, num_units=num_units))

        kernel_cache[key] = kernel.compile(thread)

    kernel_cache[key](mat)

class LWTANetwork(object):
    """
    Groups each layer's neurons. Only the neuron with maximum activation per group may be active.
    """

    def __init__(self, **kwargs):        
        super(LWTANetwork, self).__init__(**kwargs)
        log.info("LWTANetwork constructor")
        self.lwta_sizes = [0] # no LWTA for the inputs
    
    def add_layer(self, *args, **kwargs):
        super(LWTANetwork, self).add_layer(*args, **kwargs)
        lwta_size = kwargs.get('lwta', 0)
        self.lwta_sizes.append(lwta_size)
        
    def after_activation(self, layer_index, state, **kwargs):
        '''
        LWTA is applied AFTER the activation/transfer function has been
        applied. (see page 3 in http://www.idsia.ch/idsiareport/IDSIA-04-13.pdf)
        
        :param layer_index: the index of the current layer
        :param state: network state
        '''
        
        super(LWTANetwork, self).after_activation(layer_index, state, **kwargs)
        lwta_size = self.lwta_sizes[layer_index]
        if lwta_size > 1:
            activations = state.activations[layer_index]
            lwta(self.context, activations, lwta_size)