from __future__ import absolute_import, print_function, division

import logging

import numpy as np
import scipy.linalg as spla

import theano
import theano.tensor as T
from theano.tensor import as_tensor_variable
from theano.tensor.nlinalg import *
from theano.gof import Op, Apply
from theano.tensor import basic as tensor

GPU=True

if GPU:
    import pygpu
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath as cumath
    import pycuda.autoinit
    import skcuda.linalg as gpu_linalg
    gpu_linalg.init()


logger = logging.getLogger(__name__)

    
def inverse_using_cholesky(x):
    if GPU:
        if isinstance(x, pygpu.gpuarray.GpuArray):
            # we cannot perform inplace, so we have to copy
            x = x.copy(order='F')
            # then wrap in a pycuda array
            x_gpu = gpuarray.GPUArray(x.shape, x.dtype,  base=x, gpudata=(x.gpudata + x.offset), order='F')
            # make b_gpu a identity matrix
            b_gpu = pygpu.gpuarray.zeros(x_gpu.shape, x_gpu.dtype, order='F', context=x.context)
            eye_k = gpu_linalg._get_eye_kernel(x_gpu.dtype)
            N = x_gpu.shape[0]
            eye_k(b_gpu, slice=slice(0, N*N, N+1))
            # solve using cholesky
            gpu_linalg.cho_solve(x_gpu, b_gpu) # this needs Fortran order, it writes result to b_gpu
            return b_gpu  # we return the matrix in 'F' order... works only because it's symmetric!!
        else:
            print('Warning: Using slow GPU chol-inverse')
            x_gpu = gpuarray.to_gpu(x).reshape(x.shape, order='F')
            b_gpu = gpu_linalg.eye(x.shape[0]).reshape(x.shape, order='F')
            gpu_linalg.cho_solve(x_gpu, b_gpu)
            return b_gpu.reshape(x.shape, order='C').get()
    else:
        print('Using CPU chol-inverse')
        chol = spla.cholesky(x, lower=False)
        return spla.cho_solve((chol, False), np.eye(chol.shape[0]))
            
def log_det_using_cholesky(x):
    if GPU:
        if isinstance(x, pygpu.gpuarray.GpuArray):
            x = x.copy() # we cannot perform in-place
            x_gpu = gpuarray.GPUArray(x.shape, x.dtype,  base=x, gpudata=(x.gpudata + x.offset))
            gpu_linalg.cholesky(x_gpu)
            r = x_gpu.get()
        else:
            print('Warning: Using slow GPU log-det')
            x_gpu = gpuarray.to_gpu(x)
            gpu_linalg.cholesky(x_gpu)
            r = x_gpu.get()
    else:
        r = spla.cholesky(x, lower=False)
        
    return 2 * np.sum(np.log(np.diag(r)))


class MatrixInversePSD(Op):
    """Computes the inverse of a matrix :math:`A`.

    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.

    Notes
    -----
    When possible, the call to this op will be optimized to the call
    of ``solve``.

    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
#         x = as_tensor_variable(x)
        ctx = theano.gpuarray.basic_ops.infer_context_name(x)
        x_gpu = theano.gpuarray.basic_ops.as_gpuarray_variable(x, ctx)
        assert x.ndim == 2
        return Apply(self, [x_gpu], [x_gpu.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = inverse_using_cholesky(x).astype(x.dtype)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^{-1}}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.

        """
        x, = inputs
        xi = self(x)
        gz, = g_outputs
        # TT.dot(gz.T,xi)
        return [-T.dot(T.dot(xi, gz.T), xi).T]
        #return [-matrix_dot(xi, gz.T, xi).T]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return

            .. math:: \frac{\partial X^{-1}}{\partial X}V,

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: X^{-1} \cdot V \cdot X^{-1}.

        """
        x, = inputs
        xi = self(x)
        ev, = eval_points
        if ev is None:
            return [None]
        return [-T.dot(T.dot(xi, ev), xi)]
        #return [-matrix_dot(xi, ev, xi)]

    def infer_shape(self, node, shapes):
        return shapes

matrix_inverse_psd = MatrixInversePSD()

class LogDetPSD(Op):
    """
    Matrix log determinant. Input should be a square matrix.

    """

    __props__ = ()

    def make_node(self, x):
#         x = as_tensor_variable(x)
        ctx = theano.gpuarray.basic_ops.infer_context_name(x)
        x_gpu = theano.gpuarray.basic_ops.as_gpuarray_variable(x, ctx)
        assert x.ndim == 2
        o = T.scalar(dtype=x.dtype)
        return Apply(self, [x_gpu], [o])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.asarray(log_det_using_cholesky(x), dtype=x.dtype)
        except Exception:
            print('Failed to compute log determinant', x)
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * matrix_inverse_psd(x).T]

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return "LogDetPSD"
log_det_psd = LogDetPSD()