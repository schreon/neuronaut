'''
Created on Jul 7, 2014

@author: schreon
'''
import logging

import numpy
from reikna.algorithms.pureparallel import PureParallel
from reikna.core.signature import Parameter, Annotation

import neuro


log = logging.getLogger("rmsprop")

def rms_update(ctx, weights, gradient, squared_gradient, momentum, learn_rates, parameters):
    """ RMSPROP update kernel """
    kernel_cache, thread = ctx.kernel_cache, ctx.thread
    
    assert weights.shape == gradient.shape == squared_gradient.shape == momentum.shape
      
    key = (rms_update, weights.shape, thread._context) + tuple(parameters.values())
    if not key in kernel_cache.keys():
        log.info("compiling " + str(key))
        kernel = PureParallel(
            [
                Parameter('weights', Annotation(weights, 'io')),
                Parameter('gradient', Annotation(gradient, 'i')),
                Parameter('squared_gradient', Annotation(squared_gradient, 'io')),
                Parameter('momentum', Annotation(momentum, 'io')),
                Parameter('learn_rates', Annotation(learn_rates, 'io'))
            ],
            """
        ${weights.ctype} w = ${weights.load_same};
        ${gradient.ctype} g = ${gradient.load_same};        
        ${squared_gradient.ctype} sg = ${squared_gradient.load_same};
        ${momentum.ctype} m = ${momentum.load_same};
        ${learn_rates.ctype} lr = ${learn_rates.load_same};
        
        // mean gradient
        g = g / ${batch_size}f;
                            
        // Update the rms gradient
        sg = ${squared_gradient_factor}f*sg + (1.0f - ${squared_gradient_factor}f)*g*g;
        
        // normalize the gradient
        g = g / max(sqrt(sg), ${min_root_squared_gradient}f);
        
        // Adapt learn rate
        if (g*m > 0.0f) {
            lr *= ${reward_factor}f;
        } else {
            lr *= ${punish_factor}f;
        }        
        lr = max(${min_step_size}f, min(lr, ${max_step_size}f));
        
        // Update the momentum
        m = ${momentum_factor}f*m + (1.0f - ${momentum_factor}f)*g;
              
        // Apply momentum learning
        w += lr*m;
        
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
        
        ${weights.store_same}(w);
        ${momentum.store_same}(m);
        ${squared_gradient.store_same}(sg);
        ${learn_rates.store_same}(lr);
        """, guiding_array='weights', render_kwds=parameters)

        kernel_cache[key] = kernel.compile(thread)

    # Run kernel
    kernel_cache[key](weights, gradient, squared_gradient, momentum, learn_rates)

class RMSPropState(object):
    '''
    Holds the state belonging to RMSProp. 
    '''
    
    def __init__(self, **kwargs):        
        super(RMSPropState, self).__init__(**kwargs)
        ctx = kwargs['network'].context
        
        self.squared_gradients = []
        self.momentum = []
        self.learn_rates = []
        for gw, gb in self.gradients:
            sgw = ctx.thread.empty_like(gw)
            sgb = ctx.thread.empty_like(gb)
            self.squared_gradients.append((sgw, sgb))
            
            sgw.fill(numpy.float32(1.0))
            sgb.fill(numpy.float32(1.0))
            
            mw = ctx.thread.empty_like(gw)
            mb = ctx.thread.empty_like(gb)
            self.momentum.append((mw, mb))
            
            mw.fill(numpy.float32(0.0))
            mb.fill(numpy.float32(0.0))
            
            lrw = ctx.thread.empty_like(gw)
            lrb = ctx.thread.empty_like(gb)
            self.learn_rates.append((lrw, lrb))
            
            lrw.fill(numpy.float32(kwargs.get('ini_step_size', 0.0001)))
            lrb.fill(numpy.float32(kwargs.get('ini_step_size', 0.0001)))

class RMSProp(object):
    
    def __init__(self, **kwargs):
        super(RMSProp, self).__init__(**kwargs)
        log.info("RMSProp constructor")
        
        self.parameters = {
          'min_step_size' : numpy.float32(kwargs.get('min_step_size', 0.0000001)),
          'max_step_size' : numpy.float32(kwargs.get('max_step_size', 0.1)),
          'punish_factor' : numpy.float32(kwargs.get('punish_factor', 0.99)),
          'reward_factor' : numpy.float32(kwargs.get('reward_factor', 1.01)),
          'momentum_factor' : numpy.float32(kwargs.get('momentum_factor', 0.9)),
          'squared_gradient_factor' : numpy.float32(kwargs.get('squared_gradient_factor', 0.9)),
          'min_root_squared_gradient' : numpy.float32(kwargs.get('min_root_squared_gradient', 0.000001)),
          'l1_decay' : numpy.float32(kwargs.get('l1_decay', 0.0)),
          'l2_decay' : numpy.float32(kwargs.get('l2_decay', 0.0))
        }
        
        
        self.TrainingState = neuro.create("TrainingState", self.TrainingState, RMSPropState)
        
    def update_weights(self, state):
        super(RMSProp, self).update_weights(state)
        
        net = self.network
        ctx = self.context
        
        weights = net.weights
        gradients = state.gradients
        squared_gradients = state.squared_gradients
        learn_rates = state.learn_rates
        momentum = state.momentum
        
        self.parameters['batch_size'] = numpy.float32(state.size)
        for i in range(len(gradients)):
            w, b =  weights[i]
            gw, gb = gradients[i]
            sgw, sgb = squared_gradients[i]
            mw, mb = momentum[i]
            lrw, lrb = learn_rates[i]
            
            rms_update(ctx, w, gw, sgw, mw, lrw, self.parameters)
            rms_update(ctx, b, gb, sgb, mb, lrb, self.parameters)