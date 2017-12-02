import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from functools import reduce
from operator import mul
import numpy as np

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

def flatten(tensor, keep):
    fixed_shape = list(tensor.size())
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tensor.size()[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tensor.size()[i] for i in range(start, len(fixed_shape))]
    #print('out_shape',out_shape) 
    flat = tensor.view(out_shape) # [3360, 448]
    return flat

def reconstruct(tensor, ref, keep):
    ref_shape = list(ref.size())
    tensor_shape = list(tensor.size())
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or ref.size()[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tensor.size()[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape 
    #print('target_shape', target_shape) #[70, 48, 448]
    out = tensor.view(target_shape)
    return out

def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    #mask = mask.type('torch.FloatTensor')
    return torch.add(val, (1 - mask) * VERY_NEGATIVE_NUMBER)





