import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np





def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length)
    Output shapes, 
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    #sequence = torch.from_numpy(sequence).type('torch.FloatTensor')
    if isinstance(sequence, np.ndarray):
        sequence = Variable(torch.from_numpy(sequence).type('torch.FloatTensor'))
    #print('type:',type(sequence))
    populated = torch.sign(torch.abs(sequence))
    length = torch.sum(populated, 1)
    length = length.type('torch.IntTensor')
    mask = torch.unsqueeze(populated, -1)
    mask = mask.type('torch.FloatTensor')
    return length, mask