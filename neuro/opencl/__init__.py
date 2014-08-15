from neuro import BaseContext

from reikna import cluda


class OpenClContext(BaseContext):
    '''
    OpenCL context.
    '''
    def __init__(self): 
        self.api = cluda.ocl_api()  
        super(OpenClContext, self).__init__()

