from neuro import BaseContext
from neuro.cuda.cublas import dot

import numpy
from reikna import cluda


class CUDAContext(BaseContext):
    '''
    CUDA context.
    '''
    def __init__(self, *args, **kwargs): 
        self.api = cluda.cuda_api()  
        super(CUDAContext, self).__init__(*args, **kwargs)
        
        # work-around: 
        # thread synchronization does not involve CUBLAS.
        # cublas gets synchronized as soon as there is a memory transfer
        # thus we need a tiny array which is there just for synchronization
        self._sync_array = self.thread.array((1,), dtype=numpy.float32)
    
    def synchronize(self):
        super(CUDAContext, self).synchronize()
        # workaround for synchronization with CUBLAS
        self._sync_array.get()

    def dot(self, mat1, mat2, dest, trans_a=False, trans_b=False):
        '''
        Uses the (much faster!) CUBLAS implementation of sgemm instead of the reikna matmul kernel.
        
        :param mat1:
        :param mat2:
        :param dest:
        :param trans_a:
        :param trans_b:
        '''
        dot(mat1, mat2, dest, trans_a, trans_b)

                     
    