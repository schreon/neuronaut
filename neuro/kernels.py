'''
Created on Jul 7, 2014

@author: schreon

'''




import logging

import numpy
from reikna.algorithms.pureparallel import PureParallel
from reikna.algorithms.reduce import Reduce, predicate_sum
from reikna.cbrng import CBRNG
from reikna.core.signature import Annotation, Parameter
from reikna.core.signature import Type
from reikna.linalg.matrixmul import MatrixMul
from reikna.linalg.norm import EntrywiseNorm


log = logging.getLogger("kernels")

class KernelContext(object):
    '''
    Mixin with basic methods implemented as reikna kernels (work both for OpenCL and CUDA).
    '''
    def __init__(self, *args, **kwargs):
        super(KernelContext, self).__init__(*args, **kwargs)

    def add(self, mat1, mat2, dest):
        kernel_cache = self.kernel_cache
        thread = self.thread
        key = (self.add, mat1.dtype, mat1.shape)
    
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
            assert mat1.shape == mat2.shape == dest.shape
            kernel_delta_output = PureParallel(
                [
                    Parameter('mat1', Annotation(mat1, 'i')),
                    Parameter('mat2', Annotation(mat2, 'i')),
                    Parameter('dest', Annotation(dest, 'o'))
                ],
                """
            // Delta ( for the output layer )
            ${mat1.ctype} m1 = ${mat1.load_same};
            ${mat2.ctype} m2 = ${mat2.load_same};
            ${dest.ctype} d = m1 + m2;
            ${dest.store_same}(d);
            """, guiding_array='dest')
    
            kernel_cache[key] = kernel_delta_output.compile(thread)
    
        kernel_cache[key](mat1, mat2, dest)
    
    def sub(self, mat1, mat2, dest):
        """
        Subtract mat2 from mat1.
        ATTENTION: if a value is nan, the result will be zero.
        """
        kernel_cache = self.kernel_cache
        thread = self.thread
        key = (self.sub, mat1.dtype, mat1.shape)
    
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
            assert mat1.shape == mat2.shape == dest.shape
            kernel_delta_output = PureParallel(
                [
                    Parameter('mat1', Annotation(mat1, 'i')),
                    Parameter('mat2', Annotation(mat2, 'i')),
                    Parameter('dest', Annotation(dest, 'o'))
                ],
                """
            // Delta ( for the output layer )
            ${mat1.ctype} m1 = ${mat1.load_same};
            ${mat2.ctype} m2 = ${mat2.load_same};
            if (isnan(m1) || isnan(m2)) {
                ${dest.store_same}(0.0f);
            } else {                
                ${dest.ctype} d = m1 - m2;
                ${dest.store_same}(d);
            }
            """, guiding_array='dest')
    
            kernel_cache[key] = kernel_delta_output.compile(thread)
    
        kernel_cache[key](mat1, mat2, dest)
    
    def softplus(self, activations, bias, dest=None):
        kernel_cache, thread = self.kernel_cache, self.thread
        
        if dest is None:
            dest = activations
        
        key = (self.softplus, activations.shape, thread)
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
            a = log(1.0f + exp(a));
            
            ${dest.store_same}(a);
            """, guiding_array='activations')
    
            kernel_cache[key] = kernel.compile(thread, fast_math=True)
    
        # Run kernel
        kernel_cache[key](activations, bias, dest)
    
        return dest
    
    def softplus_derivative(self, activations, delta, dest=None):
        kernel_cache, thread = self.kernel_cache, self.thread
        
        if dest is None:
            dest = delta
        
        key = (self.softplus_derivative, activations.shape, thread)
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
            
            // the softplus function already has been applied 
            // to the activations, so wee need to apply the
            // inverse of softplus chained with logistic
            // note: logistic is the derivative of softplus
            a = min(max(-45.0f, a), 45.0f);
            a = 1.0f / (1.0f / (exp(a) - 1.0f) + 1.0f);
            d = d*a;
            
            ${dest.store_same}(d);
            """, guiding_array='activations')
    
            kernel_cache[key] = kernel.compile(thread)
    
        # Run kernel
        kernel_cache[key](activations, delta, dest)
    
    def linear(self, activations, bias, dest=None):
        kernel_cache, thread = self.kernel_cache, self.thread
        
        if dest is None:
            dest = activations
        
        key = (self.linear, activations.shape, thread)
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
    
    def linear_derivative(self, activations, delta, dest=None):
        # no need to do anything
        return delta
    
    
    def logistic(self, activations, bias, dest=None):
        kernel_cache, thread = self.kernel_cache, self.thread
        
        if dest is None:
            dest = activations
        
        key = (self.logistic, activations.shape, thread)
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
    
    def logistic_derivative(self, activations, delta, dest=None):
        kernel_cache, thread = self.kernel_cache, self.thread
        
        if dest is None:
            dest = delta
        
        key = (self.logistic_derivative, activations.shape, thread)
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
    
    
    def dot(self, mat1, mat2, dest, trans_a=False, trans_b=False):        
        kernel_cache, thread = self.kernel_cache, self.thread
        
        key = (self.dot, mat1.shape, mat2.shape, trans_a, trans_b, thread, mat1.dtype)
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
            assert mat1.dtype == mat2.dtype == dest.dtype
            kernel_cache[key] = MatrixMul(
                mat1, mat2, out_arr=dest, transposed_a=trans_a, transposed_b=trans_b).compile(thread)
    
        kernel_cache[key](dest, mat1, mat2)
    
    
    
    
    def norm(self, array, dest, order, axes=None):  
        """ Calculate the norm of an array """ 
        kernel_cache, thread = self.kernel_cache, self.thread
        key = (self.norm, array.shape, array.dtype, axes, thread)
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
            if axes is not None:        
                expected = tuple(array.shape[i] for i in range(len(array.shape)) if i not in axes)
                if len(expected) == 0:
                    expected = (1,)
                assert dest.shape == expected       
            comp = EntrywiseNorm(array, order=order, axes=axes)
            kernel_cache[key] = comp.compile(thread)
    
        kernel_cache[key](dest, array)
    
    def sum(self, array, target, axis=None):
        """ Sum an array along the specified axis """
        """ TODO: reduce is overkill if the reduced axis is small. instead, implement a simple kernel which loops along that axis """
        kernel_cache, thread = self.kernel_cache, self.thread
           
        if axis is not None and type(axis) is not tuple:
            axis = (axis,)
    
        key = (self.sum, array.shape, array.dtype, axis, thread)
        if not key in kernel_cache.keys():
            log.info("compiling " + str(key))
            expected_shape = tuple([array.shape[i] for i in range(len(array.shape)) if i not in axis])
            assert target.shape == expected_shape
            rd = Reduce(array, predicate_sum(array.dtype),
                        axes=axis if axis is not None else None)
            kernel_cache[key] = rd.compile(thread)
        # Run kernel
        kernel_cache[key](target, array)
    
        return array
    
    def copy_minibatch(self, array, indices, minibatch):
        kernel_cache, thread = self.kernel_cache, self.thread
        
        key = (self.copy_minibatch, minibatch.dtype, minibatch.shape, array.shape)
    
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
            assert minibatch.shape[0] == indices.shape[0]
            assert indices.dtype == numpy.int32
            
            dimensions = numpy.int32(len(array.shape))
            assert minibatch.shape[0] == indices.shape[0]
            kernel = PureParallel(
                [
                    Parameter('array', Annotation(array, 'i')),
                    Parameter('indices', Annotation(indices, 'i')),
                    Parameter('minibatch', Annotation(minibatch, 'o'))
                ],
                """
            SIZE_T idx = ${indices.load_idx}(${idxs[0]});
            %if dimensions > 1:
            ${minibatch.store_same}(${array.load_idx}(idx, ${idxs[1]}));
            %else:
            ${minibatch.store_same}(${array.load_idx}(idx));        
            %endif           
            """, guiding_array='minibatch', render_kwds=dict(dimensions=dimensions))
            
            kernel_cache[key] = kernel.compile(thread)
    
        kernel_cache[key](array, indices, minibatch)
    
    def normal(self, array, mean, std, seed=None):
        kernel_cache, thread = self.kernel_cache, self.thread
            
        key = (self.normal, array.shape, array.dtype, mean, std, thread)
        
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
    
            rng = CBRNG.normal_bm(Type(array.dtype, shape=array.shape), len(array.shape),  # @UndefinedVariable
                                  sampler_kwds=dict(mean=numpy.float32(mean), std=numpy.float32(std)), seed=seed)
    
            counters = thread.to_device(rng.create_counters())
    
            kernel_cache[key] = (rng.compile(thread), counters)
    
        (rng, counters) = kernel_cache[key]
    
        rng(counters, array)
    
        return array
    
    def randint(self, array, minval, maxval, seed=None):
        kernel_cache, thread = self.kernel_cache, self.thread
        key = (self.randint, array.shape, array.dtype, minval, maxval, thread)
            
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
    
            rng = CBRNG.uniform_integer(Type(array.dtype, shape=array.shape), len(array.shape),  # @UndefinedVariable
            sampler_kwds=dict(low=numpy.int32(minval), high=numpy.int32(maxval)), seed=seed)
    
            counters = thread.to_device(rng.create_counters())
    
            kernel_cache[key] = (rng.compile(thread), counters)
    
        (rng, counters) = kernel_cache[key]
    
        rng(counters, array)
    
    def uniform(self, array, minval, maxval, seed=None):
        kernel_cache, thread = self.kernel_cache, self.thread
        key = (self.uniform, array.shape, array.dtype, minval, maxval, thread)
            
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
    
            rng = CBRNG.uniform_float(Type(array.dtype, shape=array.shape), len(array.shape),  # @UndefinedVariable
            sampler_kwds=dict(low=numpy.float32(minval), high=numpy.float32(maxval)), seed=seed)
    
            counters = thread.to_device(rng.create_counters())
    
            kernel_cache[key] = (rng.compile(thread), counters)
    
        (rng, counters) = kernel_cache[key]
    
        rng(counters, array)
    
        return array
    
    def scale(self, mat, scalar):
        kernel_cache = self.kernel_cache
        scalar = numpy.float32(scalar)
        thread = self.thread
        key = (self.scale, mat.dtype, mat.shape)
    
        if key not in kernel_cache.keys():
            log.info("compiling " + str(key))
            kernel = PureParallel(
                [
                    Parameter('mat', Annotation(mat, 'io')),
                    Parameter('scalar', Annotation(scalar))
                ],
                """
            // Delta ( for the output layer )
            ${mat.ctype} m = ${mat.load_same};
            ${mat.ctype} s = ${scalar};
            m *= s;
            ${mat.store_same}(m);
            """, guiding_array='mat')
    
            kernel_cache[key] = kernel.compile(thread)
    
        kernel_cache[key](mat, scalar)
    
    def nan_to_zeros(self, array, dest=None):
        kernel_cache, thread = self.kernel_cache, self.thread
        
        if dest is None:
            dest = array
        
        key = (self.nan_to_zeros, array.shape, thread)
        if not key in kernel_cache.keys():
            log.info("compiling " + str(key))
    
            kernel = PureParallel(
                [
                    Parameter('array', Annotation(array, 'i')),
                    Parameter('dest', Annotation(dest, 'o')),
                ],
                """
            ${array.ctype} a = ${array.load_same};
            if (isnan(a)) {
                ${dest.store_same}(0.0f);
            }        
            """, guiding_array='array')
    
            kernel_cache[key] = kernel.compile(thread, fast_math=True)
    
        # Run kernel
        kernel_cache[key](array, dest)
    
        return dest
