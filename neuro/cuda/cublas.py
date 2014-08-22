'''
Created on Aug 13, 2014

@author: schreon
'''



from ctypes import c_char, c_float, c_uint, c_int, cast, POINTER
import ctypes
import logging
import numpy
import sys
log = logging.getLogger("cublas")

# load the library
if sys.platform == 'linux2':
    _libcublas_libname_list = ['libcublas.so', 'libcublas.so.4.0',
                               'libcublas.so.5.0', 'libcublas.so.5.5'
                               , 'libcublas.so.6.0']
elif sys.platform == 'darwin':
    _libcublas_libname_list = ['libcublas.dylib']
elif sys.platform == 'Windows':
    _libcublas_libname_list = ['cublas.lib']
else:
    raise RuntimeError('unsupported platform')

libcublas = None
for _libcublas_libname in reversed(_libcublas_libname_list):
    try:
        libcublas = ctypes.cdll.LoadLibrary(_libcublas_libname)
        log.info("found: %s" % _libcublas_libname)
    except OSError:
        pass
    else:
        break

# status codes
CUBLAS_STATUS_SUCCESS = 0x00000000
CUBLAS_STATUS_NOT_INITIALIZED = 0x00000001
CUBLAS_STATUS_ALLOC_FAILED = 0x00000003
CUBLAS_STATUS_INVALID_VALUE = 0x00000007
CUBLAS_STATUS_MAPPING_ERROR = 0x0000000B
CUBLAS_STATUS_EXECUTION_FAILED = 0x0000000D
CUBLAS_STATUS_INTERNAL_ERROR = 0x0000000E

cublasStatus = c_uint

# Exceptions
class CublasError(Exception):
    pass

def checkCublasStatus(status):
    if status != CUBLAS_STATUS_SUCCESS:
        raise CublasError("Internal cublas error: %i" % status)

# cublasGetError
_cublasGetError = libcublas.cublasGetError
_cublasGetError.restype = cublasStatus
_cublasGetError.argtypes = []

def cublasGetError():
    status = _cublasGetError()
    checkCublasStatus(status)

_cublasSgemm = libcublas.cublasSgemm
_cublasSgemm.restype = None
_cublasSgemm.argtypes = [c_char, c_char, c_int, c_int, c_int,
                         c_float, POINTER(c_float), c_int,
                         POINTER(c_float), c_int, c_float,
                         POINTER(c_float), c_int]
# cublasSgemm
def cublasSgemm(transa, transb, m, n, k, alpha, A,
                lda, B, ldb, beta, C, ldc):
    _cublasSgemm(transa, transb, m, n, k, alpha, A, lda,
                          B, ldb, beta, C, ldc)
    cublasGetError()

trans = {
     False : 'N',
     True : 'T'
}

# interface to pycuda
def dot(a, b, out, transa=False, transb=False, handle=None):

    assert a.dtype == b.dtype == out.dtype
    assert a.flags.c_contiguous == b.flags.c_contiguous == out.flags.c_contiguous

    alpha = numpy.float32(1.0)
    beta = numpy.float32(0.0)

    if transb:
        m, k = b.shape
    else:
        k, m = b.shape

    if transa:
        l, n = a.shape
    else:
        n, l = a.shape

    if l != k:
        raise ValueError('objects are not aligned')
            
    lda = max(1, b.shape[1])
    ldb = max(1, a.shape[1])
    ldc = max(1, m)
    
    ap = cast(int(a.gpudata), POINTER(c_float))
    bp = cast(int(b.gpudata), POINTER(c_float))
    outp = cast(int(out.gpudata), POINTER(c_float))

    cublasSgemm(trans[transb],
                trans[transa],
                m,
                n,
                k,
                alpha,
                bp,
                lda,
                ap,
                ldb,
                beta,
                outp,
                ldc)
