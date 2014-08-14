from neuro import BaseContext

from reikna import cluda


class OpenClContext(BaseContext):
    '''
    OpenCL context.
    '''
    def __init__(self, num_workers=10): 
        self.api = cluda.ocl_api()  
        super(OpenClContext, self).__init__(num_workers=num_workers)

