import torch
import numpy as np
# from math import prod

def tensor_times_matrix_fwd(tensor, matrix, return_saved_tensors=False):

    """ttm times matrix forward

    parameters
    ----------
    tensor : tensorly-torch tensor
    matrix : parameter object
        input matrix x
    return_saved_tensors : bool
        whether to return saved tensors for backprop

    returns
    -------
    output
        tensor times matrix equivalent to matrix.t@tensor.to_matrix()
        saved_tensors required for backpropr if return_saved_tensors is True
    """
    if tensor.name == 'TT':
        return tt_times_matrix_fwd(tensor, matrix, return_saved_tensors)
    elif tensor.name == 'BlockTT':
        return ttm_times_matrix_fwd(tensor, matrix, return_saved_tensors)
    elif tensor.name == 'Tucker':
        return tucker_times_matrix_fwd(tensor, matrix, return_saved_tensors)
    elif tensor.name == 'CP':
        return cp_times_matrix_fwd(tensor, matrix, return_saved_tensors)
    else:
        raise ValueError('Unknown tensor type')

def tt_times_matrix_fwd(tensor, matrix, return_saved_tensors):
    """
    This function takes the input tensor "tensor", the input matrix "matrix"
    and returns tensor times matrix as well as any extra tensors you decide to save
    for the backward pass
    """
    #Author Alvin Liu

    ndims = tensor.order
    d = int(ndims / 2)
    tt_shape = tensor.shape
    tt_ranks = tensor.rank[1:-1]
    tt_shape_row = tt_shape[:d]
    tt_shape_col = tt_shape[d:]
    tt_rows = np.prod(tt_shape_row)
    tt_cols = np.prod(tt_shape_col)
    matrix_rows = matrix.shape[0]
    matrix_cols = matrix.shape[1]
    if tt_rows is not None and matrix_rows is not None:
        if tt_rows != matrix_rows:
            raise ValueError(
                'Arguments shapes should align got %s and %s instead.' %
                ((tt_rows, tt_cols), (matrix_rows, matrix_cols)))

    # Matrix: M * K, tensor: M * N = (i_0, i_1, ..., i_d-1) * (j_0, j_1, ..., j_d-1)
    # The shape of data is 1 * i_0 * (i_1, i_2, ..., i_d-1, K)
    data = matrix
    data = data.reshape(1, tt_shape_row[0], -1)
    saved_tensors = [matrix] if return_saved_tensors else None

    for k in range(d):
        # The shape of data is r_k * i_k * (i_k+1, ..., i_d-1, K)
        # The shape of curr_core (core_k) is r_k * i_k * r_k+1
        # After einsum() the shape of data is r_k+1 * (i_k+1, ..., i_d-1, K)
        curr_core = tensor.factors[k]
        data = torch.einsum('ria, rib->ba', [data, curr_core])

        if k < d - 1:
            # After reshape the data, the shape is r_k+1 * i_k+1 * (i_k+2, ..., i_d-1, K)
            data = data.reshape(tt_ranks[k], tt_shape_row[k + 1], -1)

        saved_tensors.append(data) if return_saved_tensors else None

    # Now the shape of data is r_d * K
    for k in range(d):
        # The shape of data is r_d+k * (K, j_0, ..., j_k-1)
        # The shape of curr_core (core_d+k) is r_d+k * j_k * r_d+k+1
        # After einsum() the shape of data is r_d+k+1 * (K, j_0, ..., j_k-1) * j_k
        curr_core = tensor.factors[k + d]
        data = torch.einsum('ra, rjb->baj', [data, curr_core])

        if k < d - 1:
            saved_tensors.append(data.reshape(data.shape[0], matrix_cols, -1)) if return_saved_tensors else None
            # After reshape the data, the shape is r_d+k+1 * (K, j_0, ..., j_k)
            data = data.reshape(tt_ranks[k + d], -1)

    # The shape of data is 1 * (K, j_0, ..., j_d-2) * j_d-1
    # The shape of output is K * (j_0, ..., j_d-1)
    output = data.reshape(matrix_cols, tt_cols)

    if return_saved_tensors:
        return output, saved_tensors
    else:
        return output

def ttm_times_matrix_fwd(tensor, matrix, return_saved_tensors=False):
    """ttm times matrix forward

    parameters
    ----------
    tensor : blocktt
        ttm tensorized weight matrix
    matrix : parameter object
        input matrix x

    returns
    -------
    output
        tensor times matrix
        equivalent to matrix.t@tensor.to_matrix()
    saved_tensors
        tensorized input matrix
    """
    #Author Angela Chen

    # Prepare tensor shape
    shape_x = matrix.T.shape
    tensorized_shape_x, tensorized_shape_y = tensor.tensorized_shape

    num_batch = shape_x[0]
    order = len(tensor.factors)

    # Reshape transpose of input matrix to input tensor
    input_tensor = torch.reshape(matrix.T, (shape_x[0], ) + tensorized_shape_x)

    # Compute left partial sum
    # saved_tensor[k+1] = x.T * G1 * ... * Gk
    # saved_tensor[k+1].shape: num_batch * (i1*...*ik-1) * ik * (jk+1*...*jd) * rk
    saved_tensors = []
    current_i_dim = 1
    saved_tensors.append(
        input_tensor.reshape(num_batch, current_i_dim, 1, -1, tensor.rank[0]))

    for k in range(order):
        current_tensor = saved_tensors[k]
        saved_tensors.append(
            torch.einsum(
                'aibcd,dbef->aiecf',
                current_tensor.reshape(num_batch, current_i_dim,
                                       tensorized_shape_x[k], -1,
                                       tensor.rank[k]), tensor.factors[k]))
        current_i_dim *= tensorized_shape_y[k]

    # Forward Pass
    # y[i1,...,id] = sum_j1_..._jd G1[i1,j1] * G2[i2,j2] * ... * Gd[id,jd] * x[j1,...,jd]
    output = saved_tensors[order].reshape(num_batch, -1)
    
    if return_saved_tensors:
        return output, saved_tensors
    else:
        return output

def tucker_times_matrix_fwd(tensor, matrix, return_saved_tensors=False):
    """Tucker tensor times matrix forward pass"""
    #Author: Zi Yang
   
    core = tensor.core

    N = int(len(tensor.factors) / 2)

    size = [x.shape[0] for x in tensor.factors]

    out_shape = [matrix.shape[1], np.prod(size[N:])]

    output = (matrix.T).reshape([matrix.shape[1]] + size[:N])

    for i in range(N):
        output = torch.tensordot(output,
                                 tensor.factors[i],
                                 dims=[[1], [0]])

    output = torch.tensordot(output,
                             tensor.core,
                             dims=[list(range(1, N + 1)),
                                   list(range(N))])

    for i in range(N):
        output = torch.tensordot(output,
                                 tensor.factors[i + N],
                                 dims=[[1], [1]])
    output = output.reshape(out_shape)

    saved_tensors = [matrix] if return_saved_tensors else None

    if return_saved_tensors:
        return output, saved_tensors
    else:
        return output


def cp_times_matrix_fwd(tensor, matrix, return_saved_tensors=False):
    """
    Multiplies a tensorly CP tensorized matrix and an input matrix
    """
    # Author: Christian Lee
    
    saved_tensors = []
    order = len(tensor.tensorized_shape[0])
    
    # tensorize the input
    output = matrix.reshape(tensor.tensorized_shape[0] + 
                            (matrix.shape[1],))
    saved_tensors.append(output)
    
    # forward propagate with input factors
    output = torch.einsum('a...n,ar->...nr', output, tensor.factors[0])
    saved_tensors.append(output)
    for factor in tensor.factors[1:order]:
        output = torch.einsum('a...nr,ar->...nr', output, factor)
        saved_tensors.append(output)
    
    # forward propagate with output factors
    for factor in tensor.factors[order:tensor.order-1]:
        output = torch.einsum('n...r,ar->n...ar', output, factor)
        saved_tensors.append(output)
    output = torch.einsum('n...r,ar->n...a', output, tensor.factors[-1])
    
    # vectorize the output
    output = output.reshape((matrix.shape[1], tensor.shape[1]))
    
    if return_saved_tensors:
        return output, saved_tensors
    else:
        return output