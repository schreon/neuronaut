from reikna import cluda

from neuro import BaseContext
from neuro.cuda.cublas import dot

class CUDAContext(BaseContext):
    '''
    CUDA context.
    '''
    def __init__(self, *args, **kwargs): 
        self.api = cluda.cuda_api()  
        super(CUDAContext, self).__init__(*args, **kwargs)

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

                     
    