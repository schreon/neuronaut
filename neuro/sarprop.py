import logging

import numpy
from reikna.algorithms.pureparallel import PureParallel
from reikna.core.signature import Parameter, Annotation

import neuro


log = logging.getLogger("sarprop")


def sarprop_kernel(ctx, weights, gradient, last_gradient, step_sizes, noise, parameters):    
    """ SARPROP update kernel """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread

    assert weights.shape == gradient.shape == last_gradient.shape == step_sizes.shape
      
    key = (sarprop_kernel, weights.shape, thread._context) + tuple(parameters.values())
    if not key in kernel_cache.keys():
        logging.info("compiling " + str(key))
        kernel = PureParallel(
            [
                Parameter('weights', Annotation(weights, 'io')),
                Parameter('gradient', Annotation(gradient, 'i')),
                Parameter('last_gradient', Annotation(last_gradient, 'io')),
                Parameter('step_sizes', Annotation(step_sizes, 'io')),
                Parameter('noise', Annotation(step_sizes, 'i'))
            ],
            """
        ${weights.ctype} w = ${weights.load_same};
        ${gradient.ctype} g = ${gradient.load_same};
        ${last_gradient.ctype} lg = ${last_gradient.load_same};
        ${step_sizes.ctype} s = ${step_sizes.load_same};
        
        ${noise.ctype} n = ${noise.load_same};
        n = fabs(n);
    
        // Adapt step size
        if (g * lg > 0.0f) {            
            s = min(${reward_factor}f * s, ${max_step_size}f); 
            
            // Apply update
            if (g < 0.0f) {
                w = w - s*n;
            }             
            if (g > 0.0f) {
                w = w + s*n;
            } 
        } else {
            // punish step size
            s = max(${punish_factor}f * s, ${min_step_size}f);
        }

        // If l1 weight decay is greater zero, apply it
        % if l1_decay > 0.0:
        if (w > 0.0f) {
            w = max(0.0f, w - ${l1_decay}f);
        }
        if (w < 0.0f) {
            w = min(0.0f, w + ${l1_decay}f);
        }        
        % endif;
 
        // If l2 weight decay is greater zero, apply it
        % if l2_decay > 0.0:
        w *= ${1.0 - l2_decay}f;
        % endif;
               
        // Save last gradient
        lg = g;
            
        ${weights.store_same}(w);
        ${last_gradient.store_same}(lg);
        ${step_sizes.store_same}(s);
        """, guiding_array='weights', render_kwds=parameters)

        kernel_cache[key] = kernel.compile(thread)

    # Run kernel
    kernel_cache[key](weights, gradient, last_gradient, step_sizes, noise)

class SARPROPState(object):
    '''
    Holds the state belonging to SARPROP. 
    '''
    
    def __init__(self, **kwargs):        
        super(SARPROPState, self).__init__(**kwargs)
        ctx = kwargs['network'].context
                
        self.last_gradients = []
        self.step_sizes = []
        self.noise = []
        for gw, gb in self.gradients:
            lgw = ctx.thread.empty_like(gw)
            lgb = ctx.thread.empty_like(gb)
            self.last_gradients.append((lgw, lgb))
            
            lgw.fill(numpy.float32(0.0))
            lgb.fill(numpy.float32(0.0))
            
            sw = ctx.thread.empty_like(gw)
            sb = ctx.thread.empty_like(gb)
            self.step_sizes.append((sw, sb))
            
            nw = ctx.thread.empty_like(gw)
            nb = ctx.thread.empty_like(gb)
            self.noise.append((nw, nb))

            sw.fill(numpy.float32(kwargs.get('ini_step_size', 0.0001)))
            sb.fill(numpy.float32(kwargs.get('ini_step_size', 0.0001)))
            
class SARPROP(object):
    def __init__(self, **kwargs):
        super(SARPROP, self).__init__(**kwargs)
        log.info("SARPROP constructor")
              
        self.parameters = {
          'min_step_size' : numpy.float32(kwargs.get('min_step_size', 0.0000001)),
          'max_step_size' : numpy.float32(kwargs.get('max_step_size', 0.01)),
          'punish_factor' : numpy.float32(kwargs.get('punish_factor', 0.5)),
          'reward_factor' : numpy.float32(kwargs.get('reward_factor', 1.2)),
          'max_noise' : numpy.float32(kwargs.get('max_noise', 1.0)),
          'l1_decay' : numpy.float32(kwargs.get('l1_decay', 0.0)),
          'l2_decay' : numpy.float32(kwargs.get('l2_decay', 0.0))
        }
        
        self.TrainingState = neuro.create("TrainingState", self.TrainingState, SARPROPState)
        
    def update_weights(self, state):
        super(SARPROP, self).update_weights(state)
        
        net = self.network
        ctx = self.context
        
        weights = net.weights
        gradients = state.gradients
        last_gradients = state.last_gradients
        step_sizes = state.step_sizes
        noise = state.noise
        
        max_noise = self.parameters['max_noise']
        
        for i in range(len(gradients)):
            w, b =  weights[i]
            gw, gb = gradients[i]
            lgw, lgb = last_gradients[i]
            sgw, sgb = step_sizes[i]
            nw, nb = noise[i]
            ctx.uniform(nw, 0.0, max_noise, self.network.seed)
            ctx.uniform(nb, 0.0, max_noise, self.network.seed)
            sarprop_kernel(ctx, w, gw, lgw, sgw, nw, self.parameters)
            sarprop_kernel(ctx, b, gb, lgb, sgb, nb, self.parameters)
        
            