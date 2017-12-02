import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import collections
from util import blocks
from util.general import flatten, reconstruct, exp_mask
import DenseNet
import math

class DIIN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, config, seq_length, emb_dim, hidden_dim, emb_train, embeddings = None, pred_size = 3, context_seq_len = None, query_seq_len = None, dropout_rate = 0.0):
        super(DIIN, self).__init__()

        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length
        self.pred_size = pred_size 
        self.context_seq_len = context_seq_len
        self.query_seq_len = query_seq_len
        self.dropout_rate = dropout_rate
        self.config = config

        self.char_emb_cnn = nn.Conv2d(8, 100, (1, 5), stride=(1, 1), padding=0, bias=True)
        self.interaction_cnn = nn.Conv2d(448, int(448 * config.dense_net_first_scale_down_ratio), config.first_scale_down_kernel, padding=0)

        self.highway_network_linear = nn.Linear(448, 448, bias=True)
        self.self_attention_linear_p = nn.Linear(1344, 1, bias=True)
        self.self_attention_linear_h = nn.Linear(1344, 1, bias=True)

        self.fuse_gate_linear_p1 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_p2 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_p3 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_p4 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_p5 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_p6 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_h1 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_h2 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_h3 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_h4 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_h5 = nn.Linear(config.batch_size*448, 448, bias=True)
        self.fuse_gate_linear_h6 = nn.Linear(config.batch_size*448, 448, bias=True)

        self.final_linear = nn.Linear(5616, 3, bias=True)
        self.test_linear = nn.Linear(308736, 3, bias=True)

        if embeddings is not None:
            self.emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
            self.emb.weight.data.copy_(torch.from_numpy(embeddings).type('torch.LongTensor'))
            self.emb.weight.requires_grad = True

        self.char_emb_init = nn.Embedding(config.char_vocab_size, config.char_emb_size)
        self.char_emb_init.weight.requires_grad = False

        self.dense_net = DenseNet(134, config.dense_net_growth_rate, config.dense_net_transition_rate, config.dense_net_layers, config.dense_net_kernel_size)

    def dropout_rate_decay(self, global_step, decay_rate=0.997):
        p = 1 - 1 * decay_rate ** (global_step / 10000)
        self.dropout_rate = p

    def forward(self, premise_x, hypothesis_x, \
                pre_pos, hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match):
        prem_seq_lengths, prem_mask = blocks.length(premise_x)  # mask [N, L , 1]
        hyp_seq_lengths, hyp_mask = blocks.length(hypothesis_x)    	

        premise_in = F.dropout(self.emb(premise_x), p = self.dropout_rate,  training=self.training)
        hypothesis_in = F.dropout(self.emb(hypothesis_x), p = self.dropout_rate,  training=self.training)

        conv_pre, conv_hyp = self.char_emb(premise_char_vectors, hypothesis_char_vectors)

        premise_in = torch.cat([premise_in, conv_pre], 2) #[70, 48, 300], [70, 48, 100] --> [70,48,400]
        hypothesis_in = torch.cat([hypothesis_in, conv_hyp], 2)

        premise_in = torch.cat([premise_in, pre_pos], 2) # 70*48*447
        hypothesis_in = torch.cat([hypothesis_in, hyp_pos], 2)

        premise_exact_match = torch.unsqueeze(premise_exact_match,2) #70*48*1
        premise_in = torch.cat([premise_in, premise_exact_match], 2) #70*48*448
        hypothesis_exact_match = torch.unsqueeze(hypothesis_exact_match,2)
        hypothesis_in = torch.cat([hypothesis_in, hypothesis_exact_match], 2) #70*48*448
        

        premise_in = highway_network(self.highway_network_linear, premise_in, self.config.highway_num_layers, True, wd=self.config.wd, is_train = self.training)    
        hypothesis_in = highway_network(self.highway_network_linear, hypothesis_in, self.config.highway_num_layers, True, wd=self.config.wd, is_train = self.training)

        pre = premise_in  #[70, 48, 448]
        hyp = hypothesis_in
        
        for i in range(self.config.self_att_enc_layers):
            pre = self_attention_layer(self.self_attention_linear_p, self.fuse_gate_linear_p1, self.fuse_gate_linear_p2, self.fuse_gate_linear_p3, self.fuse_gate_linear_p4, self.fuse_gate_linear_p5, self.fuse_gate_linear_p6, self.config, self.training, pre, input_drop_prob=self.dropout_rate, p_mask=prem_mask) # [N, len, dim]    
            hyp = self_attention_layer(self.self_attention_linear_h, self.fuse_gate_linear_h1, self.fuse_gate_linear_h2, self.fuse_gate_linear_h3, self.fuse_gate_linear_h4, self.fuse_gate_linear_h5, self.fuse_gate_linear_h6, self.config, self.training, hyp, input_drop_prob=self.dropout_rate, p_mask=prem_mask)

        bi_att_mx = bi_attention_mx(self.config, self.training, pre, hyp, p_mask=prem_mask, h_mask=hyp_mask) # [N, PL, HL] 70,448,48,48

        bi_att_mx = F.dropout(bi_att_mx, p=self.dropout_rate, training=self.training)

        fm = self.interaction_cnn(bi_att_mx) # [70, 134, 48, 48]
  
        if self.config.first_scale_down_layer_relu:
            fm = F.relu(fm)

        premise_final = self.dense_net(fm)

        premise_final = premise_final.view(self.config.batch_size, -1)

        logits = linear(self.final_linear, [premise_final], self.pred_size ,True, bias_start=0.0, squeeze=False, wd=self.config.wd, input_drop_prob=self.config.keep_rate,
                                is_train=self.training)

        return logits

    def char_emb(self, premise_char, hypothesis_char):
        input_shape = premise_char.size()
        bs = premise_char.size(0)
        seq_len = premise_char.size(1)
        word_len = premise_char.size(2)

        premise_char = premise_char.view(-1, word_len) # (N*seq_len, word_len)
        char_pre = self.char_emb_init(premise_char)#.type('torch.LongTensor')) # (N*seq_len, word_len, embd_size)

        char_pre = char_pre.view(bs, -1, seq_len, word_len)

        hypothesis_char = hypothesis_char.view(-1, word_len)
        char_hyp = self.char_emb_init(hypothesis_char)#.type('torch.LongTensor'))
        char_hyp = char_hyp.view(bs, -1, seq_len, word_len)

        filter_sizes = list(map(int, self.config.out_channel_dims.split(','))) #[100]
        heights = list(map(int, self.config.filter_heights.split(',')))#[5]
        assert sum(filter_sizes) == self.config.char_out_size, (filter_sizes, self.config.char_out_size)    	

        def multi_conv1d(char_pre, filter_sizes, heights):
            assert len(filter_sizes) == len(heights)
            outs = []
            for filter_size, height in zip(filter_sizes, heights):
                if filter_size == 0:
                    continue
                char_pre = F.dropout2d(char_pre, p=self.dropout_rate, training=self.training) #[70, 48, 16, 8] --
                cnn_pre = self.char_emb_cnn(char_pre) #[70, 100, 48, 12]
                out = torch.max(F.relu(cnn_pre), 3)[0]  #[70, 100, 48]
                outs.append(out)
            concat_out = torch.cat(outs, 2) #[70, 100, 48]

            return concat_out


        conv_pre = multi_conv1d(char_pre, filter_sizes, heights) # [70*100*48]
        conv_hyp = multi_conv1d(char_hyp, filter_sizes, heights) # [70*100*48]

        conv_pre = conv_pre.view(-1, self.sequence_length, self.config.char_out_size)
        conv_hyp = conv_hyp.view(-1, self.sequence_length, self.config.char_out_size) # [70*48*100]

        return conv_pre, conv_hyp
   	
def linear(linear_layer, data_in, output_size, bias, bias_start=0.0, squeeze=False, wd=0.0, input_drop_prob=0, is_train = None):
    flat_datas = [flatten(data, 1) for data in data_in]

    assert is_train is not None
    flat_datas = [F.dropout(data, p=input_drop_prob, training=is_train) for data in flat_datas]

    total_data_size = sum([data.size()[1] for data in flat_datas])

    if len(flat_datas) > 1:
        flat_datas = torch.cat(flat_datas, 1)
    else:
        flat_datas = flat_datas[0]
    flat_out = linear_layer(flat_datas)

    out = reconstruct(flat_out, data_in[0], 1)

    if squeeze:
        out = torch.squeeze(out, len(list(data_in[0].size()))-1)

    return out

def highway_network(highway_network_linear, data_in, num_layers, bias, bias_start=0.0, wd=0.0, input_drop_prob=0, is_train=None, output_size = None):  		
    def highway_layer(highway_network_linear, data_in, bias, bias_start=0.0, wd=0.0, input_drop_prob=0, is_train = None, output_size = None):
        if output_size is not None:
            d = output_size
        else:
            d = data_in.size()[-1]
        trans = linear(highway_network_linear, [data_in], d, bias, bias_start=bias_start, wd=wd, input_drop_prob=input_drop_prob, is_train = is_train)
        trans = F.relu(trans)
        gate = linear(highway_network_linear, [data_in], d, bias, bias_start=bias_start, wd=wd, input_drop_prob=input_drop_prob, is_train = is_train)
        gate = F.sigmoid(gate)
        if d != data_in.size()[-1]:
            data_in = linear(highway_network_linear, [data_in], d, bias, bias_start=bias_start, wd=wd, input_drop_prob=input_drop_prob, is_train = is_train)
        out = gate * trans + (1 - gate) * data_in
        return out

    prev = data_in
    for layer_idx in range(num_layers):
        cur = highway_layer(highway_network_linear, prev, bias, bias_start=bias_start, wd=wd, 
            input_drop_prob=input_drop_prob, is_train=is_train, output_size = output_size)
        prev = cur
    return cur



def self_attention(linear_layer, config, is_train, p, p_mask=None): #[N, L, 2d]
    PL = p.size()[1]
    dim = p.size()[-1]
    p_aug_1 = torch.unsqueeze(p, 2).repeat(1,1,PL,1)
    p_aug_2 = torch.unsqueeze(p, 1).repeat(1, PL, 1, 1)

    if p_mask is None:
        ph_mask = None
    else:
        p_mask_aug_1 = torch.unsqueeze(p_mask, 2).repeat(1, 1, PL, 1).data.numpy().any(axis=3)
        p_mask_aug_2 = torch.unsqueeze(p_mask, 1).repeat(1, PL, 1, 1).data.numpy().any(axis=3)
        self_mask = Variable(torch.from_numpy((p_mask_aug_1 & p_mask_aug_2).astype(float)).type('torch.FloatTensor'), requires_grad=False)
        if config.cuda:
            self_mask = self_mask.cuda()

    h_logits = get_logits(linear_layer, [p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                          is_train=is_train, func=config.self_att_logit_func)  # [N, PL, HL]
    self_att = softsel(p_aug_2, h_logits) 
    return self_att

def self_attention_layer(self_attention_layer, fuse_gate_linear1, fuse_gate_linear2, fuse_gate_linear3, fuse_gate_linear4, fuse_gate_linear5, fuse_gate_linear6, config, is_train, p, input_drop_prob, p_mask=None):
    PL = p.size()[1]
    self_att = self_attention(self_attention_layer, config, is_train, p, p_mask=p_mask)

    p0 = fuse_gate(fuse_gate_linear1, fuse_gate_linear2, fuse_gate_linear3, fuse_gate_linear4, fuse_gate_linear5, fuse_gate_linear6, config, is_train, p, self_att, input_drop_prob)
    
    return p0

def fuse_gate(fuse_gate_linear1, fuse_gate_linear2, fuse_gate_linear3, fuse_gate_linear4, fuse_gate_linear5, fuse_gate_linear6, config, is_train, lhs, rhs, input_drop_prob):
    dim = list(lhs.size())[-1]
    lhs_1 = linear(fuse_gate_linear1, lhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
    rhs_1 = linear(fuse_gate_linear2, rhs, dim ,True, bias_start=0.0, squeeze=False, wd=0.0, input_drop_prob=input_drop_prob, is_train=is_train)
    if config.self_att_fuse_gate_residual_conn and config.self_att_fuse_gate_relu_z:
        z = F.relu(lhs_1 + rhs_1)
    else:
        z = F.tanh(lhs_1 + rhs_1)
    lhs_2 = linear(fuse_gate_linear3, lhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
    rhs_2 = linear(fuse_gate_linear4, rhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
    f = F.sigmoid(lhs_2 + rhs_2)

    if config.two_gate_fuse_gate:
        lhs_3 = linear(fuse_gate_linear5, lhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
        rhs_3 = linear(fuse_gate_linear6, rhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
        f2 = F.sigmoid(lhs_3 + rhs_3)
        out = f * lhs + f2 * z
    else:   
        out = f * lhs + (1 - f) * z

    return out


def double_linear_logits(args, size, bias, bias_start=0.0, mask=None, wd=0.0, input_drop_prob=0.0, is_train=None):
	first = torch.tanh(linear(args, size, bias, bias_start=bias_start,
		wd=wd, input_drop_prob=input_drop_prob, is_train=is_train))
	second = linear(first, 1, bias, bias_start=bias_start, squeeze=True,
		wd=wd, input_drop_prob=input_drop_prob, is_train=is_train)
	if mask is not None:
		second = exp_mask(second, mask)
	return second


def linear_logits(linear_layer, args, bias, bias_start=0.0, mask=None, wd=0.0, input_drop_prob=0.0, is_train=None):
	logits = linear(linear_layer, args, 1, bias, bias_start=bias_start, squeeze=True,
		wd=wd, input_drop_prob=input_drop_prob, is_train=is_train)
	if mask is not None:
		logits = exp_mask(logits, mask)
	return logits


def sum_logits(args, mask=None):
    rank = len(args[0].size())
    logits = sum(torch.sum(arg, rank-1) for arg in args)
    if mask is not None:
        logits = exp_mask(logits, mask)
    return logits


def get_logits(linear_layer, args, size, bias, bias_start=0.0, mask=None, wd=0.0, input_drop_prob=0.0, is_train=None, func=None):
    if func is None:
        func = "sum"
    if func == 'sum':
        return sum_logits(args, mask=mask)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, mask=mask, wd=wd, input_drop_prob=input_drop_prob,
                             is_train=is_train)
    elif func == 'double':
        return double_linear_logits(args, size, bias, bias_start=bias_start, mask=mask, wd=wd, input_drop_prob=input_drop_prob,
                                    is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask)
    elif func == 'scaled_dot':
        assert len(args) == 2
        dim = args[0].get_shape().as_list()[-1]
        arg = args[0] * args[1]
        arg = arg / tf.sqrt(tf.constant(dim, dtype=tf.float32))
        return sum_logits([arg], mask=mask)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, mask=mask, wd=wd, input_drop_prob=input_drop_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, wd=wd, input_drop_prob=input_drop_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits(linear_layer, [args[0], args[1], new_arg], bias, bias_start=bias_start, mask=mask, wd=wd, input_drop_prob=input_drop_prob,
                             is_train=is_train)
    else:
        raise Exception()

def softsel(target, logits, mask=None):
    a = softmax(logits, mask=mask)
    target_rank = len(target.size())
    out = torch.sum(torch.unsqueeze(a, -1) * target, target_rank - 2)
    return out

def softmax(logits, mask=None):
    if mask is not None:
        logits = exp_mask(logits, mask)
    flat_logits = flatten(logits, 1)
    flat_out = F.softmax(flat_logits)
    out = reconstruct(flat_out, logits, 1)

    return out

def bi_attention_mx(config, is_train, p, h, p_mask=None, h_mask=None): #[N, L, 2d]
    PL = p.size()[1] 
    HL = h.size()[1]
    p_aug = torch.unsqueeze(p, 2).repeat(1,1,HL,1)
    h_aug = torch.unsqueeze(h, 1).repeat(1,PL,1,1) #[N, PL, HL, 2d]

    h_logits = p_aug * h_aug # [70,48,48,448]
    h_logits = h_logits.view(config.batch_size, -1, PL, HL)
    return h_logits 

class Dense_net_block(nn.Module):
    def __init__(self, outChannels, growth_rate, kernel_size):
        super(Dense_net_block, self).__init__()
        self.conv = nn.Conv2d(outChannels, growth_rate, kernel_size=kernel_size, bias=False, padding=1)
        #print('block',outChannels, growth_rate)

    def forward(self, x):
        ft = F.relu(self.conv(x))
        #print("ft", ft.size())
        out = torch.cat((x, ft), dim=1)
        return out

class Dense_net_transition(nn.Module):
    def __init__(self, nChannels, outChannels):
        super(Dense_net_transition, self).__init__()
        self.conv = nn.Conv2d(nChannels, outChannels, kernel_size=1, bias=False)
        #print('trans',nChannels, outChannels)

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, (2,2), (2,2), padding=0)
        #print('trans',out.size())
        return out
