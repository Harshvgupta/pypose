import math
import torch


def vec2skew(v):
    """Batch Skew Matrix"""
    assert v.shape[-1] == 3, "Last dim should be 3"
    shape, v = v.shape, v.view(-1,3)
    S = torch.zeros(v.shape[:-1]+(3,3), device=v.device, dtype=v.dtype)
    S[:,0,1], S[:,0,2] = -v[:,2],  v[:,1]
    S[:,1,0], S[:,1,2] =  v[:,2], -v[:,0]
    S[:,2,0], S[:,2,1] = -v[:,1],  v[:,0]
    return S.view(shape[:-1]+(3,3))


def cumops_(v, dim, ops):
    L = v.shape[dim]
    assert dim != -1 or dim != v.shape[-1], "Invalid dim"
    for i in torch.pow(2, torch.arange(math.log2(L)+1, device=v.device, dtype=torch.int64)):
        index = torch.arange(i, L, device=v.device, dtype=torch.int64)
        v.index_copy_(dim, index, ops(v.index_select(dim, index), v.index_select(dim, index-i)))
    return v


def cumsum_(v, dim):
    return cumops_(v, dim, lambda a, b : a + b)


def cummul_(v, dim):
    return cumops_(v, dim, lambda a, b : a * b)


def cumprod_(v, dim):
    return cumops_(v, dim, lambda a, b : a @ b)


def cumops(v, dim, ops):
    return cumops_(v.clone(), dim, ops)


def cumsum(v, dim):
    return cumops(v, dim, lambda a, b : a + b)


def cummul(v, dim):
    return cumops(v, dim, lambda a, b : a * b)


def cumprod(v, dim):
    return cumops(v, dim, lambda a, b : a @ b)